"""F5 v3: preserve locked fold ownership and immutable experiment records."""
import json
from unittest.mock import patch

import pytest

from regime_lab.data.collector import MockGenerator, collect, run_dossier
from regime_lab.data.splits import assign_splits_by_id, bind_splits
from regime_lab.util import atomic_create_bytes, atomic_write_json, read_json


def fixture_run(path, ids=(0, 1, 2)):
    collect(MockGenerator(), ids, "registry", path, workers=1, log=lambda *_: None)


def records(path):
    return {str(p.relative_to(path)): p.read_bytes() for kind in ("raw", "gold", "grades")
            for p in (path / kind).glob("*.json")}


def test_legacy_split_binding_preserves_bytes_in_fresh_outputs(tmp_path):
    run = tmp_path / "run"
    fixture_run(run)
    manifest = read_json(run / "manifest.json")
    manifest.pop("split_policy")
    atomic_write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Legacy"):
        bind_splits(run, [0, 1, 2], out_dir=tmp_path / "out1")
    source = tmp_path / "locked.json"
    original = b'{"2": "test", "0": "train", "1": "val"}\n'
    source.write_bytes(original)
    before = records(run)
    first = bind_splits(run, [0, 1, 2], source=source, out_dir=tmp_path / "out1")
    second = bind_splits(run, [2, 1, 0], out_dir=tmp_path / "out2")
    assert first == second == {"0": "train", "1": "val", "2": "test"}
    for path in [source, run/"splits.json", tmp_path/"out1/splits.json", tmp_path/"out2/splits.json"]:
        assert path.read_bytes() == original
    assert records(run) == before
    assert len(read_json(run/"manifest.json")["splits_sha256"]) == 64


def test_new_split_policy_membership_independent_and_conflicts_refused(tmp_path):
    a, b = assign_splits_by_id(range(500)), assign_splits_by_id(range(499))
    assert all(a[key] == value for key, value in b.items())
    run = tmp_path / "run"
    fixture_run(run)
    bind_splits(run, [0, 1, 2])
    with pytest.raises(ValueError, match="population"):
        bind_splits(run, [0, 1])
    with pytest.raises(ValueError, match="population"):
        bind_splits(run, [0, 1, 2, 3])
    conflict = tmp_path / "conflict.json"
    conflict.write_text('{"0":"test","1":"test","2":"test"}')
    with pytest.raises(ValueError, match="conflicts"):
        bind_splits(run, [0, 1, 2], source=conflict)
    with pytest.raises(ValueError, match="new dataset"):
        collect(MockGenerator(), [3], "registry", run, workers=1)


def test_split_digest_detects_tamper_without_rebinding(tmp_path):
    fixture_run(tmp_path)
    bind_splits(tmp_path, [0, 1, 2])
    manifest = (tmp_path/"manifest.json").read_bytes()
    (tmp_path/"splits.json").write_text('{"0":"test","1":"test","2":"test"}')
    with pytest.raises(ValueError, match="SHA-256"):
        bind_splits(tmp_path, [0, 1, 2])
    assert (tmp_path/"manifest.json").read_bytes() == manifest


def test_missing_grade_repaired_without_model_call_or_raw_write(tmp_path):
    fixture_run(tmp_path)
    original_receipt = read_json(tmp_path/"manifest.json")
    expected = read_json(tmp_path/"grades/1.json")
    (tmp_path/"grades/1.json").unlink()
    before = records(tmp_path)
    with patch("regime_lab.data.collector.run_dossier", side_effect=AssertionError("must not regenerate")):
        result = collect(MockGenerator(), [0, 1, 2], "registry", tmp_path, workers=1)
    after = records(tmp_path)
    assert all(after[key] == value for key, value in before.items())
    repaired = read_json(tmp_path/"grades/1.json")
    assert repaired["labels"] == expected["labels"]
    assert repaired["repair_provenance"]["grader_version"] == expected["grader_version"]
    assert result["n_grades_repaired"] == 1
    assert result["n_collected"] == 0
    saved = read_json(tmp_path/"manifest.json")
    assert saved["n_collected"] == original_receipt["n_collected"]
    assert saved["started"] == original_receipt["started"]
    assert saved["resume_history"][-1]["n_grades_repaired"] == 1


@pytest.mark.parametrize("kind", ["gold", "raw"])
def test_missing_authoritative_record_fails_without_overwrite(tmp_path, kind):
    fixture_run(tmp_path)
    (tmp_path/kind/"1.json").unlink()
    before = records(tmp_path)
    with patch("regime_lab.data.collector.run_dossier", side_effect=AssertionError("must not regenerate")):
        with pytest.raises(ValueError):
            collect(MockGenerator(), [0, 1, 2], "registry", tmp_path, workers=1)
    assert records(tmp_path) == before


def test_changed_settings_and_legacy_mock_fingerprints_fail_closed(tmp_path):
    fixture_run(tmp_path)
    before = records(tmp_path)
    with pytest.raises(ValueError, match="configuration mismatch"):
        collect(MockGenerator(base_err=.9), [0, 1, 2], "registry", tmp_path)
    manifest = read_json(tmp_path/"manifest.json")
    manifest["config_fp"] = "legacy-mock-fingerprint"
    atomic_write_json(tmp_path/"manifest.json", manifest)
    with pytest.raises(ValueError, match="legacy mock"):
        collect(MockGenerator(), [0, 1, 2], "registry", tmp_path)
    assert records(tmp_path) == before


def test_mismatched_grade_turn_ids_are_not_silently_skipped(tmp_path):
    fixture_run(tmp_path)
    grade = read_json(tmp_path/"grades/1.json")
    grade["labels"][0]["turn"] = 2
    atomic_write_json(tmp_path/"grades/1.json", grade)
    before = records(tmp_path)
    with pytest.raises(ValueError, match="Grade identity"):
        collect(MockGenerator(), [0, 1, 2], "registry", tmp_path)
    assert records(tmp_path) == before


def test_direct_writer_and_atomic_publication_never_replace(tmp_path):
    fixture_run(tmp_path)
    before = records(tmp_path)
    with pytest.raises(FileExistsError):
        run_dossier(MockGenerator(), 1, "registry", tmp_path)
    with pytest.raises(FileExistsError):
        atomic_create_bytes(tmp_path/"raw/1.json", b"replacement")
    assert records(tmp_path) == before
