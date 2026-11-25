"""Small shared utilities: atomic JSON IO, hashing, seeds, numerics."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp  # re-exported

__all__ = ["atomic_write_json", "read_json", "sha256_text", "derive_seed",
           "logsumexp", "softmax_log", "fingerprint"]


def atomic_write_json(path: str | Path, obj: Any) -> None:
    """Write JSON atomically (temp file + rename) so partial files never exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=False, default=_json_default)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_create_bytes(path: str | Path, data: bytes) -> None:
    """Publish a complete new file, refusing even a concurrently created destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, path)  # atomic create-if-absent; never replaces an existing record
    finally:
        os.unlink(tmp)


def atomic_create_json(path: str | Path, obj: Any) -> None:
    atomic_create_bytes(path, json.dumps(obj, indent=1, default=_json_default).encode())


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, default=_json_default))[:16]


def derive_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from arbitrary parts (stable across runs)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def softmax_log(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalised probabilities from log-weights along an axis."""
    return np.exp(a - logsumexp(a, axis=axis, keepdims=True))


def load_report(d) -> dict:
    """report.json with [F1-diag v1] full-draw diagnostics (diagnostics_full_draw.json) merged over the
    thinned ones when present. The original thinned values are kept under 'diagnostics_thinned'."""
    d = Path(d)
    rep = read_json(d / "report.json")
    fd = d / "diagnostics_full_draw.json"
    if fd.exists():
        rep["diagnostics_thinned"] = rep.get("diagnostics", {})
        full = read_json(fd)
        rep["diagnostics"] = {m: {**rep["diagnostics_thinned"].get(m, {}), **v} for m, v in full.items()}
    return rep
