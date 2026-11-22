# ms-swift 4.6.1 x megatron-core 0.16.1 两处错配的运行时垫片(.pth 注入式,不改包文件)
# 坑1 import: is_managed_by_layer_wise_optimizer 符号缺失(仅 Muon 谓词使用)
#   修法: append 兜底 def 到 megatron/core/optimizer/layer_wise_optimizer.py
# 坑2 保存: TorchDistSaveShardedStrategy() 缺 backend/version 参数
#   症状: 训练全程正常但 checkpoint 的 iter_* 目录为空!
#   修法: 本文件 + 同名 .pth 放 dist-packages, .pth 内容为: import ms_swift_mcore_compat
try:
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy as _T
    _orig_init = _T.__init__
    def _compat_init(self, backend="torch", version=1, *a, **k):
        return _orig_init(self, backend, version, *a, **k)
    _T.__init__ = _compat_init
except Exception:
    pass
