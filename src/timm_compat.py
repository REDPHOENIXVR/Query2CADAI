"""Compatibility shim ensuring ImageNetInfo import doesn’t break even if timm >=1.0 is installed."""
import types, sys
try:
    from timm.data import ImageNetInfo  # noqa
except (ImportError, ModuleNotFoundError):
    class _Stub:  # minimal placeholder to satisfy transformers
        def __init__(self, *args, **kwargs):
            pass
    mod = sys.modules.setdefault("timm.data", types.ModuleType("timm.data"))
    setattr(mod, "ImageNetInfo", _Stub)