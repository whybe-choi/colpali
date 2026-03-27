#!/bin/bash
# Training script for ColPali-style Korean Vision Document Retrieval

export WANDB_PROJECT="Ko-VDR"
export WANDB_ENTITY="whybe-choi"

# Configuration
OUTPUT_DIR="./outputs/colgemma3-ko-$(date +%Y%m%d_%H%M%S)"
RUN_NAME="colgemma3-ko-$(date +%Y%m%d_%H%M%S)"

# Run training
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 \
    scripts/configs/gemma3/train_colgemma3_model.py \
    --model-name-or-path google/gemma-3-4b-it \
    --torch-dtype bfloat16 \
    --attn-implementation flash_attention_2 \
    --dataset-name whybe-choi/ko-vdr-train-public-v1.0 \
    --train-split train \
    --query-column-name query \
    --pos-target-column-name pos \
    --max-length 256 \
    --max-visual-tokens 768 \
    --tau 0.02 \
    --loss ce \
    --peft \
    --lora-r 32 \
    --lora-alpha 32 \
    --optim paged_adamw_8bit \
    --lora-dropout 0.05 \
    --gradcache \
    --mini-batch-size 8 \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 3 \
    --per-device-train-batch-size 64 \
    --gradient-accumulation-steps 2 \
    --lr 5e-5 \
    --warmup-steps 0.05 \
    --logging-steps 1 \
    --save-steps 500 \
    --save-total-limit 2 \
    --bf16 \
    --gradient-checkpointing \
    --report-to wandb \
    --run-name "${RUN_NAME}" \
    --dataloader-num-workers 4
