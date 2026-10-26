"""LEGACY Omega variants -- reproduced for old numbers ONLY.

These are NOT the canonical definition.  Each returns a DIFFERENT set:

  hole_ratio_otsu  : global Otsu split on r = R/T  (table_final_1019.hole_mask)
                     -> behaves like "bright half vs dark half"; covers floors,
                        white walls and sky.  Used by the benchmark table's
                        Delta(hole - non-hole).  MUST be labelled when reported.
  hole_v9_dark_edge: dark spot + edge in I but not in T (hole_v9 family).
                     -> area ~1.2%, a thin band; evaluation only, NOT a region
                        target for training.
"""
import importlib.util as _ilu
import os as _os

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_spec = _ilu.spec_from_file_location("_tf", _os.path.join(_ROOT, "eval", "table_final_1019.py"))
_tf = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_tf)


def hole_ratio_otsu(I, G):
    """DEPRECATED semantics.  Do not use as the training region."""
    return _tf.hole_mask(I, G)


def hole_v9_dark_edge(*a, **k):
    """Not ported (see archive/hole_exploration/hole_v9.py).  Import is refused
    on purpose so nobody silently trains on a 1.2%-area band."""
    raise NotImplementedError(
        "hole_v9 is an evaluation band (~1.2% area); see archive/hole_exploration/.")
