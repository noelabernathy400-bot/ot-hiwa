import json
from pathlib import Path

from experiments.synthetic.core import CONFIGS, DATA, METHODS, ROOT, source_paths


def test_experiment_configuration_and_data_layout():
    spec = json.loads((ROOT / "configs/experiments.json").read_text(encoding="utf-8"))
    assert CONFIGS == tuple(tuple(pair) for pair in spec["figure1"]["configs"])
    assert METHODS == tuple(spec["figure1"]["methods"])
    assert DATA == ROOT / "data/processed"
    for groups, dimension in CONFIGS:
        path, _ = source_paths(groups, dimension, 1)
        assert path == DATA / "synthetic" / f"S{groups:02d}_D{dimension:02d}_seed01.mat"
        assert path.is_file()


def test_project_layers_are_separate():
    root_files = {path.name for path in ROOT.iterdir() if path.is_file()}
    assert not any(name.endswith((".py", ".ps1", ".bat", ".sh")) for name in root_files)
    assert (ROOT / "experiments/synthetic/run.py").is_file()
    assert (ROOT / "experiments/neural/run.py").is_file()
    assert (ROOT / "visualization/synthetic.py").is_file()
    assert (ROOT / "visualization/neural.py").is_file()
    assert (ROOT / "experiments/synthetic/matlab/generate_synthetic_baselines.m").is_file()
