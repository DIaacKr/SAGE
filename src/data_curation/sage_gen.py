import argparse
import os
import re
import json
import random
from collections import Counter
from typing import List, Dict, Optional, Tuple, Set
from copy import deepcopy
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from mesh_sys import initialize_mesh_system, MeSHSystem

MODEL_PATH = "/models/Qwen3-4B"
MAX_TOKENS = 4096
PARALLEL_N = 2
CONSISTENCY_SAMPLES = 3
COMBO_MAX_REPEAT = 1
TERM_MAX_REPEAT = 1
CORE_CATEGORIES = {'A', 'C', 'D', 'E', 'F', 'G'}
CORE_MIN_DEPTH = 3

SELECTED_QUALIFIERS = [
    "DI", "DT", "TH", "SU", "CO", "ET", "PC", "DH",
    "AE", "AD", "TU", "PK", "PD", "PO", "TO", "CI",
    "PP", "PA", "GE", "IM", "MI", "VI", "ME"
]

def extract_answer_letter(text: str) -> Optional[str]:
    if not text: return None
    m = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if m: return m.group(1).lower()
    m2 = re.search(r"(?i)(?:answer|correct option)\s*[:：is]+\s*\(?([A-Za-z])\)?\b", text)
    if m2: return m2.group(1).lower()
    return None

def get_core_info(mesh: MeSHSystem, term_list: List[str]) -> Dict:
    core_terms = set()
    categories = set()
    for term in term_list:
        ui = mesh.get_descriptor_ui_by_name(term)
        if not ui or ui not in mesh.tree_structure: continue
        tree_nums = mesh.tree_structure[ui]['tree_numbers']
        if not tree_nums: continue
        
        max_depth = 0
        term_cats = set()
        for tn in tree_nums:
            parts = tn.split('.')
            max_depth = max(max_depth, len(parts))
            term_cats.add(parts[0][0])
        
        if any(c in CORE_CATEGORIES for c in term_cats) and max_depth > CORE_MIN_DEPTH:
            core_terms.add(term)
            categories.update(c for c in term_cats if c in CORE_CATEGORIES)
            
    template = "".join(sorted(list(categories)))
    combo_key = tuple(sorted(list(core_terms)))
    return {
        "core_terms": list(core_terms),
        "template": template,
        "combo_key": combo_key
    }

class MeSHDataSynthesizer:
    def __init__(self, args):
        self.args = args
        self.total_saved = 0
        self.output_file = args.output_file
        
        os.makedirs(os.path.dirname(self.output_file) or '.', exist_ok=True)
        
        self.mesh = initialize_mesh_system(args.desc, args.qual, args.index)
        self.llm = LLM(args.model_path, gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        self.global_combo_counter = Counter()
        self.global_term_deepening_counter = Counter()
        self.few_shot_pool = []

    def load_initial_seeds(self, seed_file):
        seeds = []
        with open(seed_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    local_terms = item.get('mesh_terms_local', [])
                    info = get_core_info(self.mesh, local_terms)
                    if not info['core_terms']: continue
                    item['_core_info'] = info
                    seeds.append(item)
                    self.few_shot_pool.append(item)
                    self.global_combo_counter[info['combo_key']] += 1
                except json.JSONDecodeError:
                    continue
        if self.args.limit:
            seeds = seeds[:self.args.limit]
        return seeds

    def _save_data(self, items: List[Dict]):
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for item in items:
                self.total_saved += 1
                info = item['_core_info']
                save_item = {
                    'id': item['id'],
                    'prompt': item['question'],
                    'response': item['response'],
                    'answer': item['answer'],
                    'mesh_terms_induced': item['mesh_terms_induced'],
                    'core_metadata': {
                        'core_terms': info['core_terms'],
                        'template': info['template'],
                    },
                    'generation_type': item['generation_type'],
                    'used_qualifier': item['used_qualifier'],
                    'consistency_score': item['consistency_score'],
                }
                f.write(json.dumps(save_item, ensure_ascii=False) + "\n")

    def _select_few_shots(self) -> List[Dict]:
        candidates = self.few_shot_pool
        k = min(len(candidates), 2)
        return random.sample(candidates, k) if k > 0 else []

    def _get_qualifier_details(self, term: str) -> List[Tuple[str, str]]:
        ui = self.mesh.get_descriptor_ui_by_name(term)
        if not ui: return []
        quals_data = self.mesh.get_qualifiers_by_descriptor_ui(ui)
        results = []
        for q in quals_data:
            q_ui = q['qualifier_ui']
            try:
                q_info = self.mesh.get_qualifier_info(q_ui) 
                abbr = q_info.get('abbreviation', '').upper()
                if abbr not in SELECTED_QUALIFIERS: continue
                name = q_info.get('name', q['qualifier_name'])
                note = q_info.get('scope_note', 'No definition available.')
                results.append((name, note))
            except Exception:
                continue
        return results

    def _build_deepening_prompt(self, term: str, qual_tuple: Optional[Tuple[str, str]]) -> str:
        ui = self.mesh.get_descriptor_ui_by_name(term)
        annots = self.mesh.get_annotation_by_ui(ui)
        term_def = annots.get('scope_note', 'No definition available.')
        
        if qual_tuple:
            q_name, q_note = qual_tuple
            aspect_instruction = f"""
Focus Aspect: "{q_name}"
Aspect Definition: {q_note}
Instruction: Create a multiple-choice question that specifically tests knowledge about the "{q_name}" of "{term}".
"""
        else:
            aspect_instruction = f"Instruction: Create a detailed multiple-choice question focusing on the definition and core attributes of \"{term}\"."

        return f"""You are a medical expert.
Target Concept: "{term}"
Concept Definition: {term_def}
{aspect_instruction}

Requirements:
1. The question must be challenging and medically precise.
2. Provide 4 plausible options.
3. Include step-by-step reasoning.

Output Format:
[Question Begin]...[Question End]
[Reasoning Begin]...[Reasoning End]
[Answer Begin]\\boxed{{A}}[Answer End]
"""

    def _build_comprehensive_prompt(self, target_terms: List[str], few_shots: List[Dict]) -> str:
        fs_text = ""
        for i, fs in enumerate(few_shots, 1):
            q_text = fs.get('question', '')
            terms_str = ", ".join(fs['_core_info']['core_terms'])
            fs_text += f"Example {i} (Concepts: {terms_str}):\n{q_text}\n\n"
            
        return f"""You are a medical expert creating high-quality multiple-choice questions.

Few-shot examples:
{fs_text}

Task: Generate a NEW medical question that meaningfully integrates these concepts: {', '.join(target_terms)}

Requirements:
1. The question should require reasoning across the concepts (e.g., interaction, differential diagnosis, comorbidity).
2. Do NOT copy examples.
3. 3-5 options, only one correct.

Output Format:
[Question Begin]...[Question End]
[Reasoning Begin]...[Reasoning End]
[Answer Begin]\\boxed{{A}}[Answer End]"""

    def _run_vllm_generate(self, prompts: List[str]):
        if not prompts: return []
        sampling_params = SamplingParams(
            temperature=0.7, top_p=0.95, max_tokens=MAX_TOKENS,
            stop=["[Answer End]", "<|im_end|>"], include_stop_str_in_output=True
        )
        final_inputs = []
        for p in prompts:
            final_inputs.append(self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], 
                tokenize=False, add_generation_prompt=True, enable_thinking=False
            ))
        return self.llm.generate(final_inputs, sampling_params)

    def process_epoch_generation(self, seeds: List[Dict], epoch_idx: int) -> Tuple[List[Dict], Set[str]]:
        if not seeds: return [], set()
        
        prompts = []
        meta_mapping = {}
        
        for seed in seeds:
            info = seed['_core_info']
            terms = info['core_terms']
            
            if len(terms) == 1:
                term = terms[0]
                if self.global_term_deepening_counter[term] >= TERM_MAX_REPEAT: continue
                
                available_quals = self._get_qualifier_details(term)
                selected_quals = []
                if not available_quals:
                    selected_quals = [None] * self.args.parallel_n
                else:
                    if self.args.parallel_n <= len(available_quals):
                        selected_quals = random.sample(available_quals, self.args.parallel_n)
                    else:
                        selected_quals = random.choices(available_quals, k=self.args.parallel_n)
                
                for q_tuple in selected_quals:
                    prompt = self._build_deepening_prompt(term, q_tuple)
                    idx = len(prompts)
                    prompts.append(prompt)
                    meta_mapping[idx] = {
                        'type': 'deepening',
                        'terms': terms,
                        'qualifier_used': q_tuple[0] if q_tuple else "None"
                    }
                self.global_term_deepening_counter[term] += 1
                
            else:
                if self.global_combo_counter[info['combo_key']] > COMBO_MAX_REPEAT: continue
                
                few_shots = self._select_few_shots()
                prompt = self._build_comprehensive_prompt(terms, few_shots)
                
                for _ in range(self.args.parallel_n):
                    idx = len(prompts)
                    prompts.append(prompt)
                    meta_mapping[idx] = {
                        'type': 'comprehensive',
                        'terms': terms,
                    }

        if not prompts: return [], set()

        outputs = self._run_vllm_generate(prompts)
        
        candidates = []
        val_prompts_flat = []
        
        for i, out in enumerate(outputs):
            meta = meta_mapping[i]
            res = self._parse_output(out.outputs[0].text, meta['terms']) 
            
            if res['valid']:
                cand = res['data']
                cand.update({
                    'generation_type': meta['type'],
                    'used_qualifier': meta.get('qualifier_used'),
                })
                candidates.append(cand)
                val_text = f"Question: {cand['question']}\n\nSelect the correct answer. Output ONLY the letter in \\boxed{{}}."
                val_prompts_flat.extend([val_text] * CONSISTENCY_SAMPLES)
        if not candidates: return [], set()

        val_outputs = self._run_vllm_generate(val_prompts_flat)
        
        valid_results = []
        new_terms_pool = set()
        
        for i, cand in enumerate(candidates):
            start = i * CONSISTENCY_SAMPLES
            votes = [extract_answer_letter(o.outputs[0].text) for o in val_outputs[start : start + CONSISTENCY_SAMPLES]]
            votes = [v for v in votes if v]
            
            if not votes:
                continue
                
            most_common, count = Counter(votes).most_common(1)[0]
            consistency = count / len(votes)
            
            if most_common == cand['answer'] and consistency >= 0.66:
                cand['id'] = f"E{epoch_idx}_{self.total_saved + len(valid_results)}"
                cand['consistency_score'] = consistency
                
                info = cand['_core_info']
                self.global_combo_counter[info['combo_key']] += 1
                
                induced = set(cand['mesh_terms_induced'])
                local = set(info['core_terms'])
                discovered = local - induced
                new_terms_pool.update(discovered)
                
                valid_results.append(cand)

        if valid_results: self._save_data(valid_results)
        
        return valid_results, new_terms_pool

    def _parse_output(self, raw_output: str, induced_core_terms: List[str]) -> Dict:
        text = raw_output
        q_m = re.search(r"\[Question Begin\]\s*(.*?)\s*\[Question End\]", text, re.DOTALL)
        a_m = re.search(r"\[Answer Begin\].*?\\boxed\{\s*([A-Za-z])\s*\}?", text, re.DOTALL)
        r_m = re.search(r"\[Reasoning Begin\]\s*(.*?)\s*\[Reasoning End\]", text, re.DOTALL)
        
        if not (q_m and a_m): return {'valid': False}
        
        question = q_m.group(1).strip()
        answer = extract_answer_letter(a_m.group(0))
        reasoning = r_m.group(1).strip() if r_m else ""
        
        if not answer: return {'valid': False}

        found_terms = self.mesh.find_mesh_terms_in_text(question)
        local_info = get_core_info(self.mesh, found_terms)
        
        return {
            'valid': True,
            'data': {
                'question': question, 'answer': answer,
                'response': f"<think>{reasoning}</think> \\boxed{{{answer}}}",
                'mesh_terms_induced': list(set(induced_core_terms)),
                '_core_info': local_info,
            }
        }

    def run_pipeline(self):
        current_seeds = self.load_initial_seeds(self.args.seed_file)
        
        for epoch in range(self.args.epochs):
            if self.args.target_count and self.total_saved >= self.args.target_count: break
            
            random.shuffle(current_seeds)
            batch_size = self.args.generation_batch_size
            next_round_candidates = [] 
            epoch_new_terms = set()
            
            for i in tqdm(range(0, len(current_seeds), batch_size), desc=f"Epoch {epoch+1}"):
                batch_seeds = current_seeds[i : i + batch_size]
                valid_items, new_terms = self.process_epoch_generation(batch_seeds, epoch)
                
                next_round_candidates.extend(valid_items)
                epoch_new_terms.update(new_terms)
                
                if self.args.target_count and self.total_saved >= self.args.target_count: break
            
            new_seeds = deepcopy(next_round_candidates)
            proto_seeds = []
            for term in epoch_new_terms:
                if self.global_term_deepening_counter[term] < TERM_MAX_REPEAT:
                    info = get_core_info(self.mesh, [term])
                    if info['core_terms']:
                        proto_seeds.append({
                            '_core_info': info,
                        })
            
            new_seeds.extend(proto_seeds)
            current_seeds = new_seeds
            
            for item in next_round_candidates:
                self.few_shot_pool.append(item)
            
            if not current_seeds: break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--desc", default="data/desc2025.jsonl")
    parser.add_argument("--qual", default="data/qual2025.jsonl")
    parser.add_argument("--index", default="data/mesh_indices.pkl")
    parser.add_argument("--output_file", default="final_dataset.jsonl")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--parallel_n", type=int, default=PARALLEL_N)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generation_batch_size", type=int, default=32)
    parser.add_argument("--target_count", type=int, default=None)
    
    args = parser.parse_args()
    synthesizer = MeSHDataSynthesizer(args)
    synthesizer.run_pipeline()

if __name__ == "__main__":
    main()
