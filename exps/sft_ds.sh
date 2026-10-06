#!/bin/bash

train_dataset_name="dataset_path" 
model_name="Qwen/Qwen2.5-7B-Instruct"  
seed=42  
lr=1e-5
epochs=5
global_batch_size=16
per_device_batch_size=1
gpu_count=2
uid="$(date +%Y%m%d_%H%M%S)"
output_dir="outputs"
exp_name="sft"
custom_run_name=""  

gradient_checkpointing=true
use_flash_attention_2=true
use_liger=true
bf16=true
offload_optimizer=false 
offload_param=false

nnodes=1  
node_rank=0  
master_addr="localhost"  
master_port=$((10000 + RANDOM % 55000))
config_file="exps/configs/deepspeed_zero3.yaml"  

do_eval=true  
eval_split_ratio=0.01  
eval_strategy="steps"  
eval_steps=0.1  

logging_steps=1
report_to="swanlab" #

save_strategy="no"
save_steps=0.6
push_to_hub=false
save_only_model=true

weight_decay=1e-4
warmup_ratio=0.05
lr_scheduler_type="cosine"
adam_beta1=0.9
adam_beta2=0.95

method="normal"
use_data_collator="false"
dataset_text_field="text"
block_size=4000

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) seed="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --global_batch_size) global_batch_size="$2"; shift 2 ;;
        --weight_decay) weight_decay="$2"; shift 2 ;;
        --train_dataset_name) train_dataset_name="$2"; shift 2 ;;
        --dataset_text_field) dataset_text_field="$2"; shift 2 ;;
        --uid) uid="$2"; shift 2 ;;
        --per_device_batch_size) per_device_batch_size="$2"; shift 2 ;;
        --gpu_count) gpu_count="$2"; shift 2 ;;
        --output_dir) output_dir="$2"; shift 2 ;;
        --exp_name) exp_name="$2"; shift 2 ;;
        --custom_run_name) custom_run_name="$2"; shift 2 ;;
        --gradient_checkpointing) gradient_checkpointing="$2"; shift 2 ;;
        --use_flash_attention_2) use_flash_attention_2="$2"; shift 2 ;;
        --use_liger) use_liger="$2"; shift 2 ;;
        --model_name) model_name="$2"; shift 2 ;;
        --master_port) master_port="$2"; shift 2 ;;
        --config_file) config_file="$2"; shift 2 ;;
        --nnodes) nnodes="$2"; shift 2 ;;
        --node_rank) node_rank="$2"; shift 2 ;;
        --master_addr) master_addr="$2"; shift 2 ;;
        --warmup_ratio) warmup_ratio="$2"; shift 2 ;;
        --logging_steps) logging_steps="$2"; shift 2 ;;
        --save_strategy) save_strategy="$2"; shift 2 ;;
        --save_steps) save_steps="$2"; shift 2 ;;
        --lr_scheduler_type) lr_scheduler_type="$2"; shift 2 ;;
        --adam_beta1) adam_beta1="$2"; shift 2 ;;
        --adam_beta2) adam_beta2="$2"; shift 2 ;;
        --push_to_hub) push_to_hub="$2"; shift 2 ;;
        --save_only_model)  save_only_model="$2";   shift 2 ;;
        --report_to) report_to="$2"; shift 2 ;;
        --bf16) bf16="$2"; shift 2 ;;
        --do_eval) do_eval="$2"; shift 2 ;;
        --eval_split_ratio) eval_split_ratio="$2"; shift 2 ;;
        --eval_strategy) eval_strategy="$2"; shift 2 ;;
        --eval_steps) eval_steps="$2"; shift 2 ;;
        --offload_optimizer) offload_optimizer="$2"; shift 2 ;;
        --offload_param) offload_param="$2"; shift 2 ;;
        --method) method="$2"; shift 2 ;;
        --use_data_collator) use_data_collator="$2"; shift 2 ;;
        --block_size) block_size="$2"; shift 2 ;;
        *) 
            echo "Warning: Unknown parameter $1, skipping" >&2
            shift 1
            ;;
    esac
done

run_config_file=$config_file
if [ "$offload_optimizer" = true ] || [ "$offload_param" = true ]; then
    echo "CPU offload is enabled. Generating temporary DeepSpeed config."
    temp_config_path=$(dirname "$config_file")/temp_$(basename "$config_file")
    
    cp "$config_file" "$temp_config_path"
    
    if [ "$offload_optimizer" = true ]; then
        echo "Offloading optimizer to CPU."
        sed -i 's/offload_optimizer_device: none/offload_optimizer_device: cpu/' "$temp_config_path"
    fi
    
    if [ "$offload_param" = true ]; then
        echo "Offloading parameters to CPU."
        sed -i 's/offload_param_device: none/offload_param_device: cpu/' "$temp_config_path"
    fi
    
    run_config_file=$temp_config_path
fi

grad_acc=$((global_batch_size/(gpu_count * nnodes * per_device_batch_size))) 
echo "Gradient accumulation steps: $grad_acc"

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export RANK=$node_rank
export WORLD_SIZE=$nnodes

echo "Using master address: $master_addr"
echo "Using master port: $master_port"
echo "Using config file: $run_config_file"
echo "Nodes: $nnodes, Current node rank: $node_rank"

log_dir="logs"
mkdir -p $log_dir

model_short_name=$(basename $model_name | sed 's/-Instruct//g')
dataset_short_name=$(basename $train_dataset_name | sed 's/-.*//' | sed 's/\/.*//')

params_name="$bs${global_batch_size}_lr${lr}_epoch${epochs}_${uid}"

if [ -z "$custom_run_name" ]; then
    run_name="${model_short_name}_${dataset_short_name}_${params_name}"
else
    run_name="${custom_run_name}_${params_name}"
fi

echo "Starting DeepSpeed training with $gpu_count GPUs per node, total $nnodes nodes"
echo "Global batch size: $global_batch_size, Per device batch size: $per_device_batch_size"
echo "Gradient accumulation steps: $grad_acc"
echo "Run name: $run_name"

accelerate launch \
    --config_file "$run_config_file" \
    --num_processes $gpu_count \
    --main_process_port $master_port \
    --machine_rank $node_rank \
    --num_machines $nnodes \
    src/train/sft.py \
    --seed $seed \
    --per_device_train_batch_size=${per_device_batch_size} \
    --per_device_eval_batch_size=${per_device_batch_size} \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=${epochs} \
    --train_file_path="${train_dataset_name}" \
    --dataset_text_field="${dataset_text_field}" \
    --model_name=$model_name \
    --warmup_ratio=${warmup_ratio} \
    --bf16=${bf16} \
    --do_eval=${do_eval} \
    --eval_strategy="${eval_strategy}" \
    --eval_split_ratio=${eval_split_ratio} \
    --eval_steps=${eval_steps} \
    --logging_steps=${logging_steps} \
    --save_strategy="${save_strategy}" \
    --save_steps=${save_steps} \
    --lr_scheduler_type="${lr_scheduler_type}" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=${adam_beta1} \
    --adam_beta2=${adam_beta2} \
    --output_dir="${output_dir}/${exp_name}/${run_name}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=${save_only_model} \
    --gradient_checkpointing=${gradient_checkpointing} \
    --report_to="${report_to}" \
    --use_flash_attention_2=${use_flash_attention_2} \
    --use_liger=${use_liger} \
    --method="${method}" \
    --use_data_collator="${use_data_collator}" \
    --block_size="${block_size}"
