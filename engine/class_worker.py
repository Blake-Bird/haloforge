"""Standalone CLASS worker used by HaloForge's crash-isolation layer."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

from engine.class_runner import _compute_direct

ARRAY_KEYS = {"k", "P", "P_by_z", "redshifts", "growth_class"}


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python -m engine.class_worker PARAMS_JSON RESULT_NPZ", file=sys.stderr)
        return 2
    params_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    try:
        params = json.loads(params_path.read_text(encoding="utf-8"))
        result = _compute_direct(params)
        arrays = {key: np.asarray(result[key]) for key in ARRAY_KEYS}
        metadata = {key: value for key, value in result.items() if key not in ARRAY_KEYS}
        with result_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays, metadata=json.dumps(metadata, default=str))
        return 0
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
