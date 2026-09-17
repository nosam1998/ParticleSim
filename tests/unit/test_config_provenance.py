from pathlib import Path

from particlesim.core.config import WarpAnalyzeConfig, load_config
from particlesim.core.provenance import build_manifest, config_hash, write_manifest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "configs"


def test_load_example_configs():
    for name in ("warp_alcubierre.yaml", "warp_natario.yaml"):
        cfg = load_config(EXAMPLES / name)
        assert isinstance(cfg, WarpAnalyzeConfig)
        assert cfg.metric.family in ("alcubierre", "natario")


def test_config_hash_is_order_independent():
    a = {"x": 1, "y": {"b": 2, "a": 1}}
    b = {"y": {"a": 1, "b": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({"x": 2, "y": {"a": 1, "b": 2}})


def test_manifest_written(tmp_path):
    cfg = WarpAnalyzeConfig()
    m = build_manifest(cfg.model_dump(), cfg.seed)
    path = write_manifest(m, tmp_path)
    assert path.exists()
    for key in ("config_hash", "git_commit", "python", "packages", "created_at"):
        assert key in m
