import json
from pathlib import Path

import yaml

from particlesim.cli import main


def test_theories_and_scenarios(capsys):
    assert main(["theories"]) == 0
    assert "gr" in capsys.readouterr().out
    assert main(["scenarios"]) == 0
    assert "warp.analyze" in capsys.readouterr().out


def test_run_warp_fast_path(tmp_path: Path, capsys):
    cfg = {
        "scenario": "warp.analyze",
        "metric": {"family": "natario"},
        "grid": {"extent": [[-8, 8]] * 3, "resolution": [8, 8, 8]},
        "analysis": {"full_stress_energy": False},
        "output": {"dir": str(tmp_path / "out"), "formats": ["npz", "json"]},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main(["run", str(path)]) == 0
    out = tmp_path / "out"
    assert (out / "fields.npz").exists() and (out / "manifest.json").exists()
    report = json.loads((out / "report.json").read_text())
    assert report["family"] == "natario"
    assert report["expansion"]["max_abs"] < 1e-12


def test_rerun_from_manifest_reproduces_report(tmp_path: Path, capsys):
    cfg = {
        "scenario": "warp.analyze",
        "metric": {"family": "alcubierre"},
        "grid": {"extent": [[-8, 8]] * 3, "resolution": [6, 6, 6]},
        "analysis": {"full_stress_energy": False},
        "output": {"dir": str(tmp_path / "a"), "formats": ["json", "h5"]},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main(["run", str(path)]) == 0
    capsys.readouterr()
    assert (tmp_path / "a" / "fields.h5").exists() and (tmp_path / "a" / "fields.xmf").exists()
    assert main(["rerun", str(tmp_path / "a" / "manifest.json"), "--out", str(tmp_path / "b")]) == 0
    ra = json.loads((tmp_path / "a" / "report.json").read_text())
    rb = json.loads((tmp_path / "b" / "report.json").read_text())
    assert ra["eulerian"] == rb["eulerian"]
