# 从 0 到 1：在 4×A800 上微调 Qwen3.8-Flash-Next 125B

> 2026-11-22，一天 10 小时，从租下机器到 125B LoRA 训练稳定跑通（GPU 利用率 92–97%）。
> 这不是教程，是实战复盘——每个问题都附报错原文（方便你搜索命中本页）、根因、解法和深挖链接。

## 一、背景：为什么这件事难

目标是微调 **Qwen3.8-Flash-Next**：125B 参数的超稀疏 MoE（每 token 激活 6B）+ 一张 **51.2B 参数的 PLE n-gram 查找表**。两个数字决定了难度：

- 主干专家层 120.8B 参数 = **241.6 GiB bf16**——显存切分是第一道生死关；
- PLE 表 95.4 GiB，**"只能沿 TP 切分"**（官方原话）——搬运它只有 Megatron（训练）和 vLLM（推理）两套官方实现。

官方 Best Practice 只给了 **8 卡**（TP2/EP4/PP2）配方。我们要在 **4 卡**上跑，且平台当前 8 卡机型缺货。于是有了这一天。

## 二、基础设施

| 项 | 值 |
|---|---|
| 机器 | compshare 华北二A 容器 · 4×A800-SXM4-80G（NVLink）· 64C/940G 内存/1TB 盘 · ¥27.78/h |
| 镜像 | cuda132_torch2130_py312（实际 Python 3.10） |
| 软件栈（最终跑通组合） | torch 2.14.1+cu130 · ms-swift 4.6.1 · megatron-core 0.16.1 · mcore-bridge 1.6.5 · transformer_engine 2.21.0（源码编译）· flash-attn 2.8.3.post2（打补丁编译）· flash-linear-attention + causal-conv1d |
| 权重 | ModelScope 直连下载 360G（131 分片），32 线程 24–43MB/s |

关键结论：**这个组合里两个轮子（TE、flash-attn）上游都没有现成的 cu13+torch2.14+py310 版本，必须自己编译**——本仓 [Releases](https://github.com/helloworldtang/qwen38-flash-next-a800/releases) 提供编译好的 flash-attn。

## 三、遇到的问题与解法（按时间顺序，报错原文可搜索）

### 阶段 A：路线试错（上午，14 轮排障后止损）

**问题 1：HF/transformers + bnb QLoRA 到底行不行？——不行，实锤。**

四道死刑，每道都有报错原文：

1. `ValueError: Some modules are dispatched on the CPU or the disk` —— accelerate 把 PLE 表按"量化后 28G"排卡，实际落卡 95.4G bf16（bnb 不量化 Embedding）。
2. 手写 device_map 把 PLE 指到 CPU 也无效 —— **父键（layers.1）会覆盖子键**（ple_embedding），hook 挂载只认块粒度。
3. CPU 落位的 PLE 权重**留在 meta 上不被物化**（transformers 的设计），forward 时 `Expected all tensors to be on the same device`。
4. swift 把该模型注册为多模态，hook 把 `input_ids` 换成 `inputs_embeds`，触发 `reverse_embedding`（逐 vocab 比较，天文数字）。

**止损判断**：官方 Best Practice 原文 "The training backend uses Megatron" 值 ¥100——第一小时就该信。→ [完整死亡证明](docs/01-为什么是Megatron-HF路线14轮实证.md)

### 阶段 B：Megatron 环境攻坚（下午前半）

**问题 2：`cannot import name 'is_managed_by_layer_wise_optimizer'`**
ms-swift 4.6.1 × megatron-core 0.16.1 接口错配（PyPI 的 mcore 止于 0.16.1，0.18+ 有此符号但要 Python≥3.12）。
解法：append 恒 False 兜底 def（仅 Muon 优化器用，LoRA 不涉及）。

**问题 3：编译 TE 报 `nccl.h: No such file or directory` → 修完又报 `cudnn.h`**
pip 版 NVIDIA 头文件不在搜索路径。解法一行：`export CPATH="$(ls -d .../dist-packages/nvidia/*/include | tr '\n' ':')$CPATH"`。

**问题 4：TE 装好了又崩 `undefined symbol: cublasLtGroupedMatrixLayoutInit_internal`**
pip 版 nvidia-cublas 13.1 太旧 → `pip install -U nvidia-cublas`。**此坑会复发**：后续任何 pip install 都可能被 torch 钉子版本连带降级，训前必查。

**问题 5：flash-attn 三种死法**
① Release 无 cu13+cp310 轮子（猜 URL 404）；② cu12torch2.8 轮子缺 `libcudart.so.12` 且 torch 2.14 删了 `c10_cuda_check_implementation` 符号；③ 源码编译报 `namespace "std" has no member "strong_ordering"`——torch 2.14 要 C++20，flash-attn setup.py 硬编码 c++17。
解法：`sed 's/c++17/c++20/g' setup.py` 后 64 并行编译 13 分钟；产物在 [Releases](https://github.com/helloworldtang/qwen38-flash-next-a800/releases)。

**问题 6：torchrun 卡 5 分钟 `DistNetworkError ... timed out after 300000ms`**
容器 hostname 不解析 → `echo "127.0.0.1 $(hostname)" >> /etc/hosts`。

### 阶段 C：显存与并行（下午中段，最大拦路虎）

**问题 7：`TEColumnParallelGroupedLinear` 实例化 OOM**
专家层 241.6G 必须切 4 刀。官方 8 卡靠 EP4；4 卡等价配方 = **TP4 / EP1 / ETP4**（`expert_tensor_parallel_size` 必须等于 TP，否则专家只被 EP 切 2 刀 = 120G/卡必爆）。PLE 用 `PLE_CPU_OFFLOAD=1` 卸到主机内存（官方机制，LoRA 不训它，仅 4% 减速）。
实测：63 GiB/卡起步，与官方 8 卡基准 63.6 同构。→ [显存数学全推导](docs/03-显存与并行公式.md)

### 阶段 D：训练期（下午后半）

**问题 8（最阴险）：130/130 步跑完，checkpoint 是空目录**
swift 的 `TorchDistSaveShardedStrategy()` 少传 `backend`/`version`（mcore 0.16.1 必传），保存瞬间 TypeError——但训练已"成功结束"，没人盯日志。
解法：`.pth` 垫片（[脚本](scripts/ms_swift_mcore_compat.py)）+ 上游 PR：[modelscope/ms-swift#10615](https://github.com/modelscope/ms-swift/pull/10615)。**判据从此改写：训练成功的标准是 checkpoint 非空，不是进度条走完。**

**问题 9：`save_total_limit` 必须 ≥2；epochs 只收 int；小数 epochs 直接拒启；截短要用 `--train_iters`**

**问题 10：速度不满 → 两连提速**
`micro_batch_size 1→2`（global 8 不变）步时 46.7s→24s（−48%）；`--train_iters` 控制总步数。

## 四、过程中的杂项坑（不致命但费时）

- 编译排障日志被自己的 `| tail` 截断，两次关键错误看不见——排障脚本必须全量落盘；
- `pkill -f` 模式匹配到自己的 ssh 命令行，三次误杀会话——用 `[r]` 括号技巧；
- 容器无 git、GitHub API 对机房 IP 限流、Release 大文件直连截断——统一走本机下载 + scp + sha256 比对；
- GPU 残留进程（kill 后显存仍占）会让下次 Megatron 起不来且报错指向并行配置——脚本开头必须清扫 compute-apps；
- 纯文本训练也被 qwen_vl_utils 强制 import torchvision。

## 五、最终结果

```
R1（2502 条垂类数据）LoRA 训练跑通：
  130 步 | loss 1.30 → 0.97 | 63–73 GiB/卡 | GPU 92–97% | 24–28s/步
checkpoint 修复后完整落盘，可复跑
```

| 资产 | 链接 |
|---|---|
| 4 卡训练脚本模板 | [scripts/train_qwen38_flash_a800.sh](scripts/train_qwen38_flash_a800.sh) |
| 保存垫片 | [scripts/ms_swift_mcore_compat.py](scripts/ms_swift_mcore_compat.py) |
| flash-attn 轮子（cu13/torch2.14/py310） | [Releases](https://github.com/helloworldtang/qwen38-flash-next-a800/releases) |
| Claude skill（agent 可装） | [skills 仓](https://github.com/helloworldtang/skills)，`npx skills add helloworldtang/skills` |
| 全天时间线 + 成本账 | [docs/04](docs/04-训练配方与踩坑时间线.md) |

**总成本约 ¥330 的学费**，换来：可复跑的 4 卡配方、两个别人没有的轮子、一个上游 PR、以及这套问题清单。10 小时的每小时买了什么，见 [docs/04](docs/04-训练配方与踩坑时间线.md)。
