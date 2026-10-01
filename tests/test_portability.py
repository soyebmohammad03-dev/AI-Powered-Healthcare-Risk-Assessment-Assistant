"""The repository must work after a fresh clone on another machine."""
import json
import subprocess

from src.data_loader import ROOT, SHA256


def test_no_machine_specific_paths_in_tracked_files():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    offenders = [f for f in files if not f.endswith((".npz", ".joblib")) and f != "tests/test_portability.py"
                 and any(s in (ROOT / f).read_text(errors="ignore") for s in ("/Users/", "/home/", "C:\\Users"))]
    assert offenders == []


def test_committed_artifacts_record_their_origin():
    for name in ("metrics", "analysis", "reliability", "shift_analysis"):
        prov = json.loads((ROOT / "artifacts" / f"{name}.json").read_text())["provenance"]
        assert prov["dataset_sha256"] == SHA256 and prov["seed"] == 42 and prov["versions"]["xgboost"]


def test_requirements_are_pinned():
    lines = [l for l in (ROOT / "requirements.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert lines and all("==" in l for l in lines)
