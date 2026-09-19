"""Environment workaround: after a macOS upgrade several Fortran-built scipy 1.15 extension modules (PROPACK, COBYLA)
fail to load ('__thread_bss' dyld error), which breaks 'import sklearn' and, through it, transformers' generation
utilities. Nothing in this project uses sklearn or those scipy solvers, so we stub them out before anything imports them."""
import sys, types, importlib.machinery as _mach
def _unavailable(*a, **k): raise RuntimeError("stubbed: unavailable in this environment (see compat.py)")
def _getattr(name):
    if name.startswith("__"): raise AttributeError(name)
    return _unavailable
for n in ["_spropack", "_dpropack", "_cpropack", "_zpropack"]:
    m = types.ModuleType(n); m.__getattr__ = _getattr; sys.modules[f"scipy.sparse.linalg._propack.{n}"] = m
sk = types.ModuleType("sklearn"); sk.__path__ = []; sk.__version__ = "0.0-stub"; sk.__spec__ = _mach.ModuleSpec("sklearn", None); skm_spec = _mach.ModuleSpec("sklearn.metrics", None); skm = types.ModuleType("sklearn.metrics"); skm.roc_curve = _unavailable; skm.__spec__ = skm_spec
sys.modules["sklearn"] = sk; sys.modules["sklearn.metrics"] = skm
so = types.ModuleType("scipy.optimize"); so.__spec__ = _mach.ModuleSpec("scipy.optimize", None); so.__path__ = []
so.linear_sum_assignment = _unavailable; so.__getattr__ = _getattr; sys.modules["scipy.optimize"] = so
