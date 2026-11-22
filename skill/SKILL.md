---
name: qwen-flash-125b-megatron-sft
description: Qwen3.8-Flash-Next 125B（MoE+51B PLE 表）在 4×A800/H20 80G 级机器上的微调全流程：环境搭建（Megatron 栈七坑速查）、显存并行公式（TP4/ETP4）、PLE 卸载 CPU、flash-attn/transformer_engine 编译修复、训练与断点排障。当要在少于 8 卡的机器上微调该模型、或遇到 swift/mcore/TE/flash-attn 版本错配报错时使用。2026-11-22 在 compshare 4×A800-SXM4-80G 全程实证。
license: MIT
agent_created: true
---

# Qwen3.8-Flash-Next 125B 少卡微调（Megatron 栈）

## 核心认知（先读，别走弯路）

1. **该模型唯一官方训练路径 = Megatron 后端**（ms-swift Best Practice 原文 "The training backend uses Megatron"）。HF/transformers 后端 + bnb QLoRA 是死路：PLE 表（51.2B 参数的 nn.Embedding）在 HF 侧无加载/摆放支持，14 轮排障实锤，别再试。
2. **专家层显存公式**：专家 120.8B 参数 × bf16 = 241.6G，**必须 4 刀切**。官方 8 卡用 EP4（63.6GiB/卡实测）；4 卡等价配方 = **TP4/EP1/ETP4**（`expert_tensor_parallel_size` 必须等于 TP，否则专家不被 TP 切分，120G/卡必 OOM，报错位置在 `TEColumnParallelGroupedLinear` 实例化）。
3. **PLE 表（95GiB bf16）只能沿 TP 切分或卸载 CPU**：设 `PLE_CPU_OFFLOAD=1`（mcore-bridge 机制，LoRA 不训它，官方实测无影响、仅 4% 减速）。
4. 少于 8 卡时不要照抄官方并行参数，按上述公式重算。

## 4 卡配方（实测 63GiB/卡、GPU 92-97%）

```bash
PLE_CPU_OFFLOAD=1 NPROC_PER_NODE=4 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron sft \
    --model /path/to/Qwen3.8-Flash-Next \
    --dataset <name> --custom_dataset_info <info.json> \
    --tuner_type lora --lora_rank 8 --lora_alpha 32 --lora_dropout 0.05 \
    --target_modules in_proj out_proj linear_proj linear_qkv \
    --tensor_model_parallel_size 4 --expert_model_parallel_size 1 \
    --expert_tensor_parallel_size 4 --pipeline_model_parallel_size 1 \
    --sequence_parallel true --moe_permute_fusion true \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --padding_free true --max_length 4096 \
    --micro_batch_size 2 --global_batch_size 8 \
    --lr 1e-4 --train_iters <N> \
    --save_total_limit 2   # megatron 强制 ≥2，=1 会启动即报错
```

- 提速要点：`micro_batch_size 2`（等效 batch 不变，累积减半）实测步时 46.7s→27.0s（−42%），显存 63→73GiB/卡，80G 卡放得下。
- 截短训练用 `--train_iters N`，**不要用小数 epochs**（megatron 后端 epochs 只收 int；epochs 会覆盖 train_iters 使进度条分母仍是全量）。

## 环境搭建顺序（每步都有坑，按序执行）

基础镜像：`cuda132_torch2130_py312`（实际 Python 3.10，装包后 torch 会被抬到 2.14.1+cu130）。

```bash
pip install ms-swift==4.6.1 "transformers==5.20.2" bitsandbytes==0.50.2
pip install mcore-bridge          # 连带装 megatron-core 0.16.1（PyPI 最高版）
pip install "flash-linear-attention>=0.5.2" causal-conv1d pybind11 ninja
pip install torchvision --no-deps # 纯文本训练也被 qwen_vl_utils 强制 import
```

### 坑 1：transformer_engine 必须源码编译（mcore-bridge import 时硬依赖）

```bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.0"          # A800=sm_80；H20=9.0
export CPATH="$(ls -d /usr/local/lib/python3.10/dist-packages/nvidia/*/include | tr '\n' ':')$CPATH"
export MAX_JOBS=64                          # 64核全开
pip install --no-build-isolation "transformer_engine[pytorch]"
```
缺 CPATH 会依次报 `nccl.h`/`cudnn.h` not found——头文件都在 pip 版 nvidia 包里，一次全加。

### 坑 2：TE 运行时 `undefined symbol: cublasLtGroupedMatrixLayoutInit_internal`

pip 版 `nvidia-cublas` 太旧 → `pip install -U nvidia-cublas`（13.1→13.10）。**注意任何后续 pip install 都可能把它连带降回去**（torch 的钉子版本），训前必查 `pip show nvidia-cublas`。

### 坑 3：swift 4.6.1 × mcore 0.16.1 两处接口错配

- import 期：`cannot import name 'is_managed_by_layer_wise_optimizer'`——该符号仅 Muon 优化器谓词用，LoRA/AdamW 恒 False。修法：append 兜底 def 到 `megatron/core/optimizer/layer_wise_optimizer.py`。
- 保存期：`TorchDistSaveShardedStrategy() missing 2 required positional arguments: 'backend' and 'version'`——**不修的话训练全程正常但 checkpoint 为空目录**。修法：新建 `dist-packages/swift_mcore_compat.py`（monkeypatch __init__ 默认 `backend='torch', version=1`）+ 同名 `.pth` 写 `import swift_mcore_compat`（.pth 在每个解释器启动时生效，覆盖 torchrun 子进程；直接改包文件会被权限/升级冲掉）。
- 注意：mcore 0.17+ 有这些符号但要 Python≥3.12（`pyproject.toml` 为准，setup.cfg classifiers 会骗人），PyPI 无 0.17-0.19。

### 坑 4：flash-attn 预编译轮子全不可用，需打 C++20 补丁源码编译

- 官方 Release 无 cu13+cp310 轮子；cu12torch2.8 轮子双重死：缺 `libcudart.so.12` + torch 2.14 删了 `c10_cuda_check_implementation` 符号。
- 源码编译前必须 `sed -i 's/c++17/c++20/g' setup.py`（4 处；torch 2.14 头文件要 `std::strong_ordering`），否则 nvcc 报 `namespace "std" has no member "strong_ordering"`。
- 下载 sdist 走 PyPI JSON API 拿直链（`https://pypi.org/pypi/flash-attn/<ver>/json`）；服务器直连 GitHub Release CDN 会截断，走本机下载 + scp。

### 坑 5：torchrun rendezvous 超时 300s

容器 hostname 不解析 → `echo "127.0.0.1 $(hostname)" >> /etc/hosts`。

### 坑 6：GPU 残留进程

kill 后 nvidia-smi 仍占显存；训练脚本开跑前必须 `nvidia-smi --query-compute-apps=pid --format=csv,noheader` 逐个 kill，否则下次 Megatron 起不来且报错指向并行配置，极难排查。

### 坑 7：Python 3.10 无 git、GitHub API 限流

大文件/GitHub 操作：本机（可挂 VPN）gh api 查资产 → 本机 curl 下载 → scp 上服务器；sha256 两端比对后再用。

## 数据注册（swift 4.6.1 新格式）

```json
[{"dataset_name": "xxx", "messages": {}, "dataset_path": "/abs/path/train.jsonl"}]
```
旧格式（`"messages": "messages"` 列名字符串）会报 `argument after ** must be a mapping`。ChatML 三段式（system/user/assistant），多数据集就并列多条。

## 训后验收冒烟

```bash
python3 -c "import mcore_bridge, flash_attn, transformer_engine; print('ok')"
pip show nvidia-cublas | grep Version   # 必须 13.10+
```
checkpoint 内 `iter_*/` 目录非空才叫保存成功（出过全程正常但目录为空的事故）。

## 排障速查

| 症状 | 根因 → 修法 |
|---|---|
| `TEColumnParallelGroupedLinear` 实例化 OOM | 专家没切 4 刀 → TP4/ETP4 |
| PLE 相关 device 错误 | 走了 HF 后端 → 换 Megatron + PLE_CPU_OFFLOAD=1 |
| 训练正常但 checkpoint 空 | 坑 3 保存错配 → .pth 垫片 |
| `save_total_limit` ValueError | 必须 ≥2 |
| 小数 epochs 报 invalid int | 用 `--train_iters` |
| nvcc 报 strong_ordering | 坑 4 的 c++20 补丁 |
| TE/bridge import 报 cublasLt 符号 | 坑 2 升级 nvidia-cublas |
| 编译日志只看到 "failed" 不见细节 | 排障脚本禁止 `| tail`，全量落盘 |
