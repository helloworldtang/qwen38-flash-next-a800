#!/usr/bin/env bash
# 4×A800-80G（sm_80，64C/940G）上的 QLoRA 环境准备
# 与 00_env_8xh20.sh 的区别：走 HF 后端，不需要 Megatron/mcore-bridge；
# 需要 bitsandbytes（NF4 量化）。fla / causal_conv1d 是该架构内核，sm_80 可编译。
set -euo pipefail

export no_proxy="${no_proxy:-},.modelscope.cn,modelscope.cn,.aliyuncs.com,mirrors.aliyun.com"
export NO_PROXY="${NO_PROXY:-},.modelscope.cn,modelscope.cn,.aliyuncs.com,mirrors.aliyun.com"
PIP="python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/"

command -v pip3 >/dev/null || {
  cat > /etc/apt/sources.list <<'SRC'
deb https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse
SRC
  apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-pip python3-venv build-essential
}

echo "=== 1. 核心框架（与 9B 验证过的版本一致，勿随意升级）==="
$PIP -U pip
$PIP "ms-swift==4.6.1" "transformers==5.20.2" "qwen_vl_utils>=0.0.14"

echo "=== 2. QLoRA 量化依赖 ==="
$PIP "bitsandbytes==0.50.2"

echo "=== 3. GDN/QSA 内核（缺了会静默退化成慢速实现，日志只有一句 falling back）==="
$PIP "flash-linear-attention>=0.5.2"
$PIP causal-conv1d || echo "[WARN] causal-conv1d 装失败，训练可继续（慢），推理务必补装"

echo "=== 4. 核对 ==="
python3 - <<'PY'
import importlib
for m in ["torch","swift","transformers","bitsandbytes","fla","causal_conv1d"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:16s} OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(f"  {m:16s} MISSING ({type(e).__name__})")
PY
python3 -c "import torch;print('cuda', torch.cuda.is_available(), torch.cuda.device_count(),'cards')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=== 5. QLoRA 旗标核对（版本若不符会在这里暴露）==="
swift sft --help 2>/dev/null | grep -E "quant_method|quant_bits|bnb_4bit|device_map" || echo "[WARN] help 里没搜到量化旗标，开跑前先 swift sft --help 人工确认"
