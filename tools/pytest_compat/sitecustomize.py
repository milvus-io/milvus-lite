"""Compatibility shims for running upstream Milvus client tests locally."""

import numpy as np

if not hasattr(np, "NaN"):
    np.NaN = np.nan
