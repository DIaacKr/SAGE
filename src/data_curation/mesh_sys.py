import json
import pickle
import os
import re
from typing import Dict, List

class MeSHSystem:
    def __init__(self):
        self.descriptor_by_ui = {}
        self.descriptor_by_name = {}
        self.term_to_descriptor = {}
        self.tree_structure = {}
        self.qualifier_by_ui = {}
        self.qualifier_by_name = {}
        self.qualifier_term_to_ui = {}
        self.qualifier_index = {}
        
    def build_descriptor_indices(self, jsonl_file: str):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    descriptor_ui = record['DescriptorUI']
                    descriptor_name = record['DescriptorName']['String']
                    
                    self.descriptor_by_ui[descriptor_ui] = record
                    self.descriptor_by_name[descriptor_name.lower()] = record
                    
                    if 'ConceptList' in record:
                        concepts = record['ConceptList']['Concept']
                        if isinstance(concepts, dict):
                            concepts = [concepts]
                        
                        for concept in concepts:
                            terms = concept['TermList']['Term']
                            if isinstance(terms, dict):
                                terms = [terms]
                            
                            for term in terms:
                                term_name = term['String'].lower()
                                self.term_to_descriptor[term_name] = descriptor_ui
                    
                    if 'TreeNumberList' in record:
                        tree_numbers = record['TreeNumberList']['TreeNumber']
                        if isinstance(tree_numbers, str):
                            tree_numbers = [tree_numbers]
                        
                        self.tree_structure[descriptor_ui] = {
                            'tree_numbers': tree_numbers,
                            'parent': self._get_parent_tree(tree_numbers),
                            'children': []
                        }
                    
                    if 'AllowableQualifiersList' in record:
                        qualifiers = record['AllowableQualifiersList']['AllowableQualifier']
                        if isinstance(qualifiers, dict):
                            qualifiers = [qualifiers]
                        
                        self.qualifier_index[descriptor_ui] = [
                            {
                                'qualifier_ui': qual['QualifierReferredTo']['QualifierUI'],
                                'qualifier_name': qual['QualifierReferredTo']['QualifierName']['String'],
                                'abbreviation': qual['Abbreviation']
                            }
                            for qual in qualifiers
                        ]
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    
        print(f"Descriptor indices built: {len(self.descriptor_by_ui)}")
    
    def build_qualifier_indices(self, jsonl_file: str):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    qualifier_ui = record['QualifierUI']
                    qualifier_name = record['QualifierName']['String']
                    
                    self.qualifier_by_ui[qualifier_ui] = record
                    self.qualifier_by_name[qualifier_name.lower()] = record
                    
                    if 'ConceptList' in record:
                        concepts = record['ConceptList']['Concept']
                        if isinstance(concepts, dict):
                            concepts = [concepts]
                        
                        for concept in concepts:
                            terms = concept['TermList']['Term']
                            if isinstance(terms, dict):
                                terms = [terms]
                            
                            for term in terms:
                                term_name = term['String'].lower()
                                self.qualifier_term_to_ui[term_name] = qualifier_ui
                                
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    
        print(f"Qualifier indices built: {len(self.qualifier_by_ui)}")
    
    def _get_parent_tree(self, tree_numbers):
        parents = []
        for tree_number in tree_numbers:
            parts = tree_number.split('.')
            if len(parts) > 1:
                parent = '.'.join(parts[:-1])
                parents.append(parent)
        return parents if parents else None
    
    def build_tree_relationships(self):
        for descriptor_ui, tree_info in self.tree_structure.items():
            if tree_info['parent']:
                for parent_tree in tree_info['parent']:
                    for potential_parent_ui, potential_parent_info in self.tree_structure.items():
                        if parent_tree in potential_parent_info['tree_numbers']:
                            if descriptor_ui not in potential_parent_info['children']:
                                potential_parent_info['children'].append(descriptor_ui)
    
    def save_indices(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'descriptor_by_ui': self.descriptor_by_ui,
                'descriptor_by_name': self.descriptor_by_name,
                'term_to_descriptor': self.term_to_descriptor,
                'tree_structure': self.tree_structure,
                'qualifier_by_ui': self.qualifier_by_ui,
                'qualifier_by_name': self.qualifier_by_name,
                'qualifier_term_to_ui': self.qualifier_term_to_ui,
                'qualifier_index': self.qualifier_index
            }, f)
        print(f"Indices saved to: {filepath}")
    
    def load_indices(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.descriptor_by_ui = data['descriptor_by_ui']
            self.descriptor_by_name = data['descriptor_by_name']
            self.term_to_descriptor = data['term_to_descriptor']
            self.tree_structure = data['tree_structure']
            self.qualifier_by_ui = data['qualifier_by_ui']
            self.qualifier_by_name = data['qualifier_by_name']
            self.qualifier_term_to_ui = data['qualifier_term_to_ui']
            self.qualifier_index = data['qualifier_index']
        print("Indices loaded.")

    def get_descriptor_ui_by_name(self, term_name: str) -> str:
        term_name_lower = term_name.lower()
        if term_name_lower in self.term_to_descriptor:
            return self.term_to_descriptor[term_name_lower]
        elif term_name_lower in self.descriptor_by_name:
            return self.descriptor_by_name[term_name_lower]['DescriptorUI']
        return None
    
    def get_qualifiers_by_descriptor_ui(self, descriptor_ui: str) -> List[Dict]:
        return self.qualifier_index.get(descriptor_ui, [])
    
    def get_qualifier_info(self, qualifier_ui: str) -> Dict:
        if qualifier_ui in self.qualifier_by_ui:
            record = self.qualifier_by_ui[qualifier_ui]
            return {
                'ui': qualifier_ui,
                'name': record['QualifierName']['String'],
                'annotation': record.get('Annotation', ''),
                'scope_note': self._get_qualifier_scope_note(record),
                'abbreviation': self._get_qualifier_abbreviation(record),
                'synonyms': self._get_qualifier_synonyms(record)
            }
        return {}

    def get_annotation_by_ui(self, descriptor_ui: str) -> Dict:
        if descriptor_ui in self.descriptor_by_ui:
            record = self.descriptor_by_ui[descriptor_ui]
            annotations = {}
            
            if 'ConceptList' in record:
                concepts = record['ConceptList']['Concept']
                if isinstance(concepts, dict):
                    concepts = [concepts]
                
                for concept in concepts:
                    if 'ScopeNote' in concept:
                        annotations['scope_note'] = concept['ScopeNote']
            
            if 'HistoryNote' in record:
                annotations['history_note'] = record['HistoryNote']
            
            if 'PublicMeSHNote' in record:
                annotations['public_mesh_note'] = record['PublicMeSHNote']
                
            return annotations
        return {}

    def get_synonyms_by_name(self, term_name: str) -> List[str]:
        descriptor_ui = self.get_descriptor_ui_by_name(term_name)
        if descriptor_ui and descriptor_ui in self.descriptor_by_ui:
            record = self.descriptor_by_ui[descriptor_ui]
            synonyms = []
            
            if 'ConceptList' in record:
                concepts = record['ConceptList']['Concept']
                if isinstance(concepts, dict):
                    concepts = [concepts]
                
                for concept in concepts:
                    terms = concept['TermList']['Term']
                    if isinstance(terms, dict):
                        terms = [terms]
                    
                    for term in terms:
                        term_str = term['String']
                        if term_str.lower() != term_name.lower():
                            synonyms.append(term_str)
            
            return list(set(synonyms))
        return []
    
    def _get_qualifier_scope_note(self, qualifier_record: Dict) -> str:
        if 'ConceptList' in qualifier_record:
            concepts = qualifier_record['ConceptList']['Concept']
            if isinstance(concepts, dict):
                concepts = [concepts]
            
            for concept in concepts:
                if 'ScopeNote' in concept:
                    return concept['ScopeNote']
        return ""
    
    def _get_qualifier_abbreviation(self, qualifier_record: Dict) -> str:
        if 'ConceptList' in qualifier_record:
            concepts = qualifier_record['ConceptList']['Concept']
            if isinstance(concepts, dict):
                concepts = [concepts]
            
            for concept in concepts:
                terms = concept['TermList']['Term']
                if isinstance(terms, dict):
                    terms = [terms]
                
                for term in terms:
                    if 'Abbreviation' in term:
                        return term['Abbreviation']
        return ""
    
    def _get_qualifier_synonyms(self, qualifier_record: Dict) -> List[str]:
        synonyms = []
        if 'ConceptList' in qualifier_record:
            concepts = qualifier_record['ConceptList']['Concept']
            if isinstance(concepts, dict):
                concepts = [concepts]
            
            main_name = qualifier_record['QualifierName']['String']
            
            for concept in concepts:
                terms = concept['TermList']['Term']
                if isinstance(terms, dict):
                    terms = [terms]
                
                for term in terms:
                    term_str = term['String']
                    if term_str.lower() != main_name.lower():
                        synonyms.append(term_str)
        
        return list(set(synonyms))
    
    def find_mesh_terms_in_text(self, text):
        def normalize_text(text):
            return re.sub(r'[^\w\s]', '', text.lower()).strip()
        
        matched_descriptors = set()
        normalized_text = ' ' + normalize_text(text) + ' '
        
        for term_name, descriptor_record in self.descriptor_by_name.items():
            descriptor_ui = descriptor_record['DescriptorUI']
            
            normalized_term = ' ' + normalize_text(term_name) + ' '
            if normalized_term in normalized_text:
                matched_descriptors.add(descriptor_ui)
                continue
                
            synonyms = self.get_synonyms_by_name(term_name)
            for synonym in synonyms:
                normalized_synonym = ' ' + normalize_text(synonym) + ' '
                if normalized_synonym in normalized_text:
                    matched_descriptors.add(descriptor_ui)
                    break
        
        result_terms = []
        for ui in matched_descriptors:
            if ui in self.descriptor_by_ui:
                term_name = self.descriptor_by_ui[ui]['DescriptorName']['String']
                result_terms.append(term_name)
                
        return result_terms
        
def initialize_mesh_system(desc_jsonl: str, qual_jsonl: str, indices_file: str = "mesh_indices.pkl"):
    if os.path.exists(indices_file):
        mesh_system = MeSHSystem()
        mesh_system.load_indices(indices_file)
        return mesh_system
    
    mesh_system = MeSHSystem()
    mesh_system.build_descriptor_indices(desc_jsonl)
    mesh_system.build_qualifier_indices(qual_jsonl)
    mesh_system.build_tree_relationships()
    mesh_system.save_indices(indices_file)
    
    return mesh_system
