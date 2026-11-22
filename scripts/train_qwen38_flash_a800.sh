#!/usr/bin/env bash
# v0.7 R1：Qwen3.8-Flash-Next 125B 垂类 LoRA SFT（Megatron 后端，4×A800 80G）
# 依据官方 Best Practice，8 卡配方改为 TP2/EP2/PP1（4 卡），PLE 卸 CPU。
# 与 HF/QLoRA 路线的差异：bf16 无量化误差，target_modules 用官方架构专属模块。
set -euo pipefail

MODEL=${MODEL:-/root/work/models/Qwen3.8-Flash-Next}
DATA=${DATA:-/root/work/data}
OUT=${OUT:-/root/work/output/125b-megatron-v0.7-r1}
DS=${DS:-chip_dft_r1}
NPROC=${NPROC:-4}

# PLE（51B N-gram 表，bf16 95GiB）卸载 CPU：LoRA 本就不更新它，官方实测无影响
export PLE_CPU_OFFLOAD=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export NPROC_PER_NODE=${NPROC}

for _p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    echo "[clean] 清理残留 GPU 进程 ${_p}"; kill -9 "${_p}" 2>/dev/null || true
done
sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

megatron sft \
    --model "${MODEL}" \
    --dataset "${DS}" \
    --custom_dataset_info "${DATA}/custom_dataset_info.json" \
    --split_dataset_ratio 0.02 \
    --num_train_epochs 1 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules in_proj out_proj linear_proj linear_qkv \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --padding_free true \
    --max_length 4096 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --lr 1e-4 \
    --enable_thinking false \
    --logging_steps 5 \
    --save_steps 200 \
    --save_total_limit 1 \
    --dataloader_num_workers 4 \
    --output_dir "${OUT}"

# R2 复用：DS=chip_dft_r2 OUT=/root/work/output/125b-megatron-v0.7-r2 bash train_r1_megatron.sh
