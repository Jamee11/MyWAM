import importlib.util
from pathlib import Path


def test_redirect_boundary():
    path = Path(__file__).resolve().parents[1] / "scripts/download_assets/download_ea_foundation_mirror.py"
    spec = importlib.util.spec_from_file_location("mirror_download", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.allowed_url("https://hf-mirror.com/api/models/a")
    assert not module.allowed_url("https://huggingface.co/a")
    assert not module.allowed_url("https://cas-bridge.xethub.hf.co/blob")
    assert module.allowed_url("https://cas-bridge.xethub.hf.co/blob", True)
    assert not module.allowed_url("https://hf-mirror.com.evil.org/a", True)
    assert not module.allowed_url("http://hf-mirror.com/a", True)


def test_parallel_ranges_preserve_prefix_resume_pieces_and_assemble_exact_bytes(tmp_path, monkeypatch):
    import contextlib
    path = Path(__file__).resolve().parents[1] / "scripts/download_assets/download_ea_foundation_mirror.py"
    spec = importlib.util.spec_from_file_location("mirror_download", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    payload = bytes(range(80))
    partial = tmp_path / "weights.partial"
    partial.write_bytes(payload[:5])
    pieces = tmp_path / "weights.partial.ranges"; pieces.mkdir()
    (pieces / "5-11.partial").write_bytes(payload[5:8])
    requested = []
    @contextlib.contextmanager
    def fake_get(session, url, *, headers, **kwargs):
        assert session.trust_env is False
        start, end = map(int, headers["Range"][6:].split("-"))
        requested.append((start, end))
        class Response:
            status_code = 206
            headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
            def iter_content(self, _):
                yield payload[start:end + 1]
        yield Response()
    monkeypatch.setattr(module, "get", fake_get)
    module.download_ranges("https://hf-mirror.com/repo/file", partial, len(payload), 4, True, block_size=7)
    assert partial.read_bytes() == payload
    assert (8, 11) in requested and (5, 11) not in requested


def test_range_response_must_match_requested_offsets(tmp_path, monkeypatch):
    import contextlib
    import pytest
    path = Path(__file__).resolve().parents[1] / "scripts/download_assets/download_ea_foundation_mirror.py"
    spec = importlib.util.spec_from_file_location("mirror_download", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    @contextlib.contextmanager
    def fake_get(*args, **kwargs):
        class Response:
            status_code = 200
            headers = {}
        yield Response()
    monkeypatch.setattr(module, "get", fake_get)
    partial = tmp_path / "weights.partial"; partial.write_bytes(b"prefix")
    with pytest.raises(RuntimeError, match="exact range"):
        module.download_ranges("https://hf-mirror.com/file", partial, 20, 2, True, block_size=7)
    assert partial.read_bytes() == b"prefix"
