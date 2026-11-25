"""Dataset-bound split ownership. Legacy studies require an explicit saved assignment."""
import hashlib
import json
import os
from pathlib import Path

from ..util import atomic_create_bytes, atomic_write_json, read_json

POLICY = "per-id-v1"


def assign_splits_by_id(ids, seed=2026):
    """Prospective 60/20/20 hash allocation; membership never enters the hash."""
    result = {}
    for did in sorted(ids):
        digest = hashlib.sha256(f"{POLICY}|{seed}|{int(did)}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        result[str(int(did))] = "train" if value < .6 else "val" if value < .8 else "test"
    return result


def _validate(data, ids):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"Duplicate split ID: {key}")
            obj[key] = value
        return obj
    splits = json.loads(data, object_pairs_hook=unique)
    expected = {str(int(i)) for i in ids}
    if not isinstance(splits, dict) or set(splits) != expected:
        raise ValueError("Canonical splits require exact dataset ID coverage; create a new dataset version for new IDs")
    if not all(value in {"train", "val", "test"} for value in splits.values()):
        raise ValueError("Unknown fold in canonical splits")
    return splits


def bind_splits(run_dir, ids, source=None, out_dir=None):
    """Reuse canonical bytes across outputs; explicit migration only for legacy data.

    New collections declare POLICY before any evaluation. Old datasets cannot be
    mistaken for new ones simply because the destination directory is empty.
    """
    root = Path(run_dir)
    ids = list(map(int, ids))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate dossier IDs")
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if set(manifest["ids"]) != set(ids):
        raise ValueError("Collection population differs from completed dataset; refusing partial evaluation")
    canonical = root / "splits.json"
    if canonical.exists():
        data = canonical.read_bytes()
        if source is not None and Path(source).read_bytes() != data:
            raise ValueError("Requested split source conflicts with the canonical bytes")
    elif source is not None:
        data = Path(source).read_bytes()
    elif manifest.get("split_policy") == POLICY and not manifest.get("splits_sha256"):
        data = json.dumps(assign_splits_by_id(ids, manifest["base_seed"]), indent=1).encode()
    else:
        raise ValueError("Legacy or incomplete split binding: provide --split-source pointing to the existing locked splits.json")
    splits = _validate(data, ids)
    digest = hashlib.sha256(data).hexdigest()
    if manifest.get("splits_sha256") not in (None, digest):
        raise ValueError("Canonical split SHA-256 mismatch")
    if manifest.get("splits") not in (None, "splits.json"):
        raise ValueError("Unsupported/conflicting canonical split location")
    target = Path(out_dir) / "splits.json" if out_dir is not None else None
    if target is not None and target.exists() and target.read_bytes() != data:
        raise ValueError("Output split assignment conflicts with the canonical dataset")
    if not canonical.exists():
        atomic_create_bytes(canonical, data)
    if manifest.get("splits_sha256") != digest:
        manifest.update(splits="splits.json", splits_sha256=digest)
        if source is not None:
            manifest["split_binding_source"] = os.path.relpath(Path(source).resolve())  # no absolute paths in records
        atomic_write_json(manifest_path, manifest)
    if target is not None and not target.exists():
        atomic_create_bytes(target, data)
    return splits


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bind canonical dataset splits without refitting or modifying experiment records")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--source", help="Existing locked split file for explicit legacy migration")
    args = parser.parse_args()
    root = Path(args.dir)
    ids = []
    for path in sorted((root / "raw").glob("*.json")):
        raw = read_json(path)
        if not raw.get("complete") or str(raw["dossier_id"]) != path.stem:
            raise ValueError(f"Incomplete or misidentified raw record: {path}")
        ids.append(raw["dossier_id"])
    result = bind_splits(root, ids, source=args.source)
    print(f"Bound {len(result)} dossiers to {root / 'splits.json'}; SHA-256 {read_json(root / 'manifest.json')['splits_sha256']}")
