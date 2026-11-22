# qwen-flash-125b-toolkit

Qwen3.8-Flash-Next 125B（MoE 125B + PLE 51B）在 **4×A800-80G** 级机器上的微调工具箱。
2026-11-22 在 compshare 华北二A（4×A800-SXM4-80G, 64C/940G, CUDA 13.2）全程实证。

## 为什么需要这个仓库

官方 Best Practice 的训练路径是 **8 卡 Megatron**（TP2/EP4/PP2）。在只有 4 卡 80G 时：
预编译轮子（cu13 + torch 2.14 + py3.10 组合）上游全部缺失，且 swift 4.6.1 × megatron-core 0.16.1
存在两处接口错配。本仓库把一天的环境攻坚沉淀为可直接使用的资产。

## 内容

| 项 | 说明 |
|---|---|
| `skill/SKILL.md` | 完整方法论：4 卡配方、七坑速查、排障表（也可放入 Claude skills 目录直接用） |
| `patches/flash-attn-cxx20.patch` | flash-attn 2.8.3.post2 在 torch≥2.14 下的 C++20 编译修复 |
| `patches/ms-swift-mcore016-compat.py` | swift 4.6.1 × mcore 0.16.1 保存崩溃垫片（训练正常但 checkpoint 为空的那个坑） |
| Release 资产 | 编译好的 wheel：`flash_attn` / `transformer_engine`（cu13+torch2.14.1+cu130+cp310+sm80） |

## 4 卡核心配方

```
TP4 / EP1 / ETP4 + PLE_CPU_OFFLOAD=1 + bf16 LoRA
实测: 63-73GiB/卡, GPU 利用率 92-97%, micro_batch=2 时 24-27s/step
```

专家层 241.6G(bf16) 必须切 4 刀——`expert_tensor_parallel_size` 必须等于 TP，
这是 4 卡能否跑起来的全部秘密。详见 skill/SKILL.md。

## 安装轮子

```bash
pip install flash_attn-2.8.3.post2+cu13torch2.14cxx20-cp310-cp310-linux_x86_64.whl
pip install transformer_engine_torch-2.21.0+cu13torch2.14.1-cp310-cp310-linux_x86_64.whl
pip install -U nvidia-cublas   # 必须 13.10+，否则 TE 运行时缺符号
```

## 适用边界

- 验证组合：CUDA 13.2 / torch 2.14.1+cu130 / Python 3.10 / sm_80(A800)；H20(sm_90) 需改 `TORCH_CUDA_ARCH_LIST="9.0"` 重编
- 上游若发布官方 cu13 轮子或 swift 修复错配，本仓库资产即完成历史使命
