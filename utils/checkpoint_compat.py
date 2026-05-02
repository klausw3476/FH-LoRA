"""
PyTorch checkpoints saved under NumPy 2 reference numpy._core.*; NumPy 1.x only has
numpy.core.*. Also strip repo `.python-packages` from sys.path when prepended via
PYTHONPATH (wrong ABI wheels shadow site-packages).

Import this module before `torch.load` on checkpoints (side effects on import).
"""

from __future__ import annotations

import importlib
import pkgutil
import sys


def drop_repo_python_packages_from_sys_path() -> None:
    drop = [
        p
        for p in list(sys.path)
        if p
        and (p.endswith(".python-packages") or "/.python-packages" in p.replace("\\", "/"))
    ]
    for p in drop:
        while p in sys.path:
            sys.path.remove(p)


def register_numpy2_core_aliases() -> None:
    import numpy as np

    if hasattr(np, "_core"):
        return

    import numpy.core as _core_pkg

    sys.modules.setdefault("numpy._core", _core_pkg)

    _subs = (
        "multiarray",
        "umath",
        "_multiarray_umath",
        "numeric",
        "defchararray",
        "_dtype_ctypes",
        "overrides",
    )
    for _sub in _subs:
        _old = f"numpy.core.{_sub}"
        _new = f"numpy._core.{_sub}"
        try:
            _mod = importlib.import_module(_old)
            sys.modules.setdefault(_new, _mod)
        except ImportError:
            pass

    # Full tree: unpicklers may reference any numpy.core.* submodule as numpy._core.*
    try:
        import numpy.core as _nc

        for _finder, _name, _ispkg in pkgutil.walk_packages(
            _nc.__path__, _nc.__name__ + "."
        ):
            if not _name.startswith("numpy.core."):
                continue
            _alias = _name.replace("numpy.core", "numpy._core", 1)
            if _alias in sys.modules:
                continue
            try:
                _mod = importlib.import_module(_name)
                sys.modules.setdefault(_alias, _mod)
            except ImportError:
                pass
    except Exception:
        pass


def install() -> None:
    drop_repo_python_packages_from_sys_path()
    register_numpy2_core_aliases()


install()
