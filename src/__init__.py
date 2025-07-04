import os, sys
_current_dir = os.path.dirname(__file__)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from .utils import ensure_startup_dirs
ensure_startup_dirs()