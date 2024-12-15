#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 T
export TOKENIZERS_PARALLELISM=true 
export HF_HOME=~/.cache/huggingface/datasets 
export PYTHONPATH=$1:$PYTHONPATH/ 
export ROOT_DIR=$2 
export MASTER_ADDR=127.0.0.1 
export MASTER_PORT=12355 
export RANK=0 
export LOCAL_RANK=0 
export WORLD_SIZE=1 
export USE_LLAVA=1 
export HF_API_KEY=$3 
python3 duo_attn/eval/efficiency/benchmark_dynamic.py --disable_wandb --max_length=1000 --sparsity=0.75 --attn_load_dir outputs_50k/ --output_dir outputs_50k_evalVQA_0.75/