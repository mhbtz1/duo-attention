#!/bin/bash

export CUDA_VISIBLE_DEVICES=4 
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
export HF_API_KEY=${$3:-$HF_API_KEY}
python3 duo_attn/train.py --disable_wandb --dataset_format menu --context_lengths_num_intervals=50 --num_steps=50000 --context_length_min=5000 --context_length_max=5000 --num_passkeys=5 --max_length=10000