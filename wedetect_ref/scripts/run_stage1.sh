#!/bin/bash


MODEL_TYPE=qwen2p5_instruct
OUTPUT_DIR=output
OUTPUT_DIR_FT=${OUTPUT_DIR}/Qwen3-VL-2B-stage1
mkdir -p ${OUTPUT_DIR_FT}


NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
PORT=${MASTER_PORT:-29513}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

torchrun --nproc_per_node 8 \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    sft.py \
    --output_dir "./log/Qwen3-VL-2B-stage1" \
    --model_name_or_path "models/Qwen3-VL-2B-Instruct" \
    --dataset_name "datasets/wedetect_ref/pretrain_data_685k.json" \
    --data_folder "datasets/" \
    --deepspeed scripts/zero2.json \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Qwen3-VL-2B-sft-stage1 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --max_grad_norm 5 \
    --dataloader_prefetch_factor 2 \
    --dataloader_num_workers 2 \
    --freeze_vision_modules true \
    --freeze_llm_modules true \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/stage1_log_node_$RANK.txt && echo "Done."

