"""Compatibility shim for older timm versions.

Ensures that both ImageNetInfo (class) and infer_imagenet_subset (function)
exist in timm.data, providing harmless stubs if missing. This prevents
ImportError in downstream packages (e.g., transformers) expecting recent timm APIs.
"""

import types
import sys

# Get or create the timm.data module
mod = sys.modules.setdefault("timm.data", types.ModuleType("timm.data"))

# Patch ImageNetInfo if missing
try:
    from timm.data import ImageNetInfo  # noqa
except (ImportError, ModuleNotFoundError, AttributeError):
    class _ImageNetInfoStub:
        """Minimal stub to satisfy consumers expecting ImageNetInfo."""
        def __init__(self, *args, **kwargs):
            pass
    setattr(mod, "ImageNetInfo", _ImageNetInfoStub)

# Patch infer_imagenet_subset if missing
if not hasattr(mod, "infer_imagenet_subset"):
    def infer_imagenet_subset(*args, **kwargs):
        """Stub for infer_imagenet_subset: does nothing, returns None."""
        return None
    setattr(mod, "infer_imagenet_subset", infer_imagenet_subset)