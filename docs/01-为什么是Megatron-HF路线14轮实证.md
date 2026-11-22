# 01 · 路线选择：为什么必须是 Megatron（HF/QLoRA 死路实证）

> 2026-11-22，compshare 4×A800-SXM4-80G，Qwen3.8-Flash-Next（125B MoE + 51B PLE 表）。
> 结论先行：**官方 Best Practice 原文 "The training backend uses Megatron"**——这句话值得先信。
> 我们不信邪用 HF 后端 + bnb QLoRA 试了 14 轮，以下是完整死亡证明，帮你省掉同样的半天。

## 模型的特殊构造

| 组件 | 规模 | 特性 |
|---|---|---|
| 主干 MoE | 125B（激活 6B） | 48 层，3/4 GDN 线性注意力 + 1/4 QSA 稀疏注意力 |
| **PLE n-gram 表** | **51.2B** | 单个 nn.Embedding，bf16 = 95.4 GiB；checkpoint 里存成 128 个 `shard_N.weight` 分片，加载时 concat |
| MTP 头 | 小 | 多 token 预测，训练不涉及 |

## HF 路线的四道死刑（逐条实证）

### 死刑 1：accelerate 规划器估不准 PLE

`device_map auto` 挂上 `BitsAndBytesConfig` 后，规划器按"量化后 28G"给 PLE 排卡，实际落卡是 95.4G bf16（bnb 不量化 Embedding）→ 必 OOM。

### 死刑 2：device_map 父键覆盖子键

手写 device_map 把 `layers.1.ple.ple_embedding` 指到 CPU、`layers.1` 整层指到 GPU——accelerate 的 hook 挂载只认块粒度，**父键存在时子键被无视**，PLE 照样被搬上 GPU。

### 死刑 3：transformers 的 meta 设计

transformers 对"允许不在 device_map 的 embedding"的设计是**留 meta**——CPU 落位的 PLE 可能根本不被物化，forward 时 `execution_device=None` 直接崩。

### 死刑 4：swift 的多模态 hook

swift 把 `qwen4_exp` 注册为 MLLM，`post_encode` hook 把 `input_ids` 转成 `inputs_embeds` 再 pop 掉——触发 PLE 的 `reverse_embedding`（逐 vocab 位置比较，[B,S,V,H] 张量，天文数字）。

## 附带收获

- 显式 device_map 生成、模型结构审计（`init_empty_weights` + 按模块形态统计参数）的方法论沉淀在 skill 里。
- PLE 在 HF 侧 forward 代码本身支持 CPU 查表（每次只搬查询行），但加载链路不给机会。

## 一句话

> 51B 的 embedding 表不是"大模型微调"的常规难题，是这个模型**架构级**的特殊件，
> 官方只给它做了 Megatron（TP 分片/CPU 卸载）和 vLLM（推理）两套搬运方案。
> 没有第三条路，别再造轮子。
