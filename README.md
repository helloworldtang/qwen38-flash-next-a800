# Qwen3.8-Flash-Next on A800

**在 4×A800-80G 上微调 Qwen3.8-Flash-Next 125B 的一切**：路线实证、显存公式、环境七坑、编译轮子、训练脚本、成本账。
2026-11-22 全程实战沉淀（compshare 华北二A · 4×A800-SXM4-80G · 64C/940G · CUDA 13.2 · torch 2.14.1+cu130 · py3.10）。

## 一分钟版

```
1. 别走 HF/QLoRA，官方唯一训练路径是 Megatron（51B PLE 表只有它搬得动）→ docs/01
2. 4 卡配方 = TP4 / EP1 / ETP4 + PLE_CPU_OFFLOAD=1（专家 241.6G 必须切 4 刀）→ docs/03
3. 环境有七个坑，按 docs/02 的顺序做，别乱序
4. 轮子直接下载 Release（flash_attn / transformer_engine，cu13+torch2.14+py310 组合上游全缺）
5. 训练成功的判据是 checkpoint 非空，不是进度条走完
```

## 内容导航

| 路径 | 内容 |
|---|---|
| [docs/01](docs/01-为什么是Megatron-HF路线14轮实证.md) | HF 后端 14 轮排障的完整死亡证明——为什么死、怎么死的 |
| [docs/02](docs/02-环境搭建七坑实录.md) | TE 编译/cublas 符号/版本错配/flash-attn cxx20/hostname/残留进程/杂项 |
| [docs/03](docs/03-显存与并行公式.md) | 专家层 4 刀切分公式、官方 8 卡 → 4 卡参数映射、实测显存与步时 |
| [docs/04](docs/04-训练配方与踩坑时间线.md) | 可抄配方、全天时间线、每小时的钱买了什么教训 |
| [scripts/](scripts/) | 训练脚本模板、swift×mcore 兼容垫片 |
| [skill/SKILL.md](skill/SKILL.md) | 同方法论打包成 Claude skill（速查表形态） |
| [Releases](https://github.com/helloworldtang/qwen38-flash-next-a800/releases) | 编译轮子 |

## Release 轮子

| 轮子 | 组合 | 说明 |
|---|---|---|
| `flash_attn 2.8.3.post2+cu13torch2.14cxx20` | cu13 / torch2.14.1 / py310 / sm80 | 含 -std=c++20 修复（上游 Release 无此组合） |
| `transformer_engine_torch 2.21.0` | cu13 / torch2.14.1 / py310 / sm80 | 官方 Release 无此组合（猜 URL 404 已核实） |

安装后务必 `pip install -U nvidia-cublas`（≥13.10），否则 TE 运行时缺符号。

## 关联

- 上游修复 PR：[modelscope/ms-swift#10615](https://github.com/modelscope/ms-swift/pull/10615)（训练正常但 checkpoint 为空的保存崩溃）
- 官方 Best Practice：[swift docs](https://swift.readthedocs.io/en/v4.5/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)（8 卡口径）

## 适用边界

- 实证组合如上；H20（sm_90）需改 `TORCH_CUDA_ARCH_LIST="9.0"` 重编两个轮子
- 上游发布 cu13 官方轮子或 swift 修复错配后，本仓相应内容自动过期
