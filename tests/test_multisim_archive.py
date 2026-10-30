import json

import pytest

from scripts.archive_multisim_sources import run


def test_independent_copy_survives_source_removal_and_detects_corruption(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "raw.h5").write_bytes(b"native replay information")
    root = tmp_path / "archive"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(roots=[dict(source=str(source), destination="native")])))
    state = root / "state.sqlite"
    run(root, plan, state, "copy", 2)
    target = root / "native/raw.h5"
    assert target.stat().st_ino != (source / "raw.h5").stat().st_ino
    assert target.stat().st_nlink == 1
    (source / "raw.h5").unlink()
    run(root, plan, state, "verify", 2)
    target.write_bytes(b"bad")
    with pytest.raises(ValueError, match="audit failed"):
        run(root, plan, state, "verify", 2)


def test_internal_symlink_is_materialized_and_cycle_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "real").write_bytes(b"replay")
    (source / "alias").symlink_to("real")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(roots=[dict(source=str(source), destination="native")])))
    root = tmp_path / "archive"
    run(root, plan, root / "state.sqlite", "copy", 2)
    assert not (root / "native/alias").is_symlink()
    assert (root / "native/alias").read_bytes() == b"replay"
    (source / "cycle").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink cycle"):
        run(tmp_path / "other", plan, tmp_path / "other/state.sqlite", "inventory", 1)


def test_external_execution_view_requires_explicit_preserved_descriptor(tmp_path):
    source, code = tmp_path / "source", tmp_path / "code"
    source.mkdir()
    code.mkdir()
    (source / "raw.h5").write_bytes(b"trajectory")
    (code / "replay.py").write_text("print('replay')\n")
    (code / "config.yml").write_text("action_type: ee\n")
    (source / "execution_view").symlink_to(code, target_is_directory=True)
    (source / "config.yml").symlink_to(code / "config.yml")
    plan = tmp_path / "plan.json"
    roots = [dict(source=str(source), destination="native")]
    plan.write_text(json.dumps(dict(roots=roots)))
    with pytest.raises(ValueError, match="explicit dependency plan"):
        run(tmp_path / "rejected", plan, tmp_path / "rejected/state.sqlite", "copy", 1)
    roots.append(dict(source=str(code), destination="replay/code"))
    descriptors = [dict(source=str(source / "execution_view"), target=str(code), kind="execution_view")]
    file_links = [dict(source=str(source / "config.yml"), target=str(code / "config.yml"))]
    plan.write_text(json.dumps(dict(roots=roots, metadata_only_links=descriptors, materialize_external_files=file_links)))
    root = tmp_path / "archive"
    run(root, plan, root / "state.sqlite", "copy", 2)
    assert not (root / "native/execution_view").exists()
    assert not (root / "native/config.yml").is_symlink()
    assert (root / "native/config.yml").read_bytes() == (code / "config.yml").read_bytes()
    assert (root / "replay/code/replay.py").read_bytes() == (code / "replay.py").read_bytes()
    saved = list((root / "provenance").glob("source_links_*.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text()) == descriptors


def test_changed_source_is_not_silently_copied(tmp_path):
    source = tmp_path / "raw.h5"
    source.write_bytes(b"first")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(roots=[dict(source=str(source), destination="raw.h5")])))
    root = tmp_path / "archive"
    run(root, plan, root / "state.sqlite", "inventory", 1)
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Source changed since inventory"):
        run(root, plan, root / "state.sqlite", "copy", 1)
    assert not (root / "raw.h5").exists()
