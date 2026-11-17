"""Validation for processed data, author references, and final artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/synthetic/author_reference/hiwa/runs"
DEFAULT_NEURAL = ROOT / "results/neural/author_reference/hiwa"
REQUIRED_RESULTS = (
    "results/synthetic/main/figures/figure1_all_six.png",
    "results/synthetic/main/per_trial_errors.csv",
    "results/synthetic/main/runs/S02_D02_seed01.npz",
    "results/neural/main/gcot/GCOT.npz",
    "results/neural/main/gcot/figures/figure2a.png",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_author_inputs(results=DEFAULT_RESULTS, neural=DEFAULT_NEURAL):
    results = Path(results)
    synthetic = sorted(results.glob("figure1[ef]_seed*.mat"))
    if len(synthetic) != 100:
        raise RuntimeError(f"Expected 100 author synthetic inputs, found {len(synthetic)}")
    records = {}
    for path in synthetic:
        data = loadmat(path, simplify_cells=True)
        required = {"XX", "YY", "Rgt", "Lx", "Ly", "S", "seed",
                    "methods", "baseline_aligned", "baseline_errors"}
        missing = required.difference(data)
        if missing:
            raise RuntimeError(f"{path.name} missing fields: {sorted(missing)}")
        methods = np.atleast_1d(data["methods"]).tolist()
        if methods != ["HiWA-SSC", "WA", "CORAL", "SA", "ICP"]:
            raise RuntimeError(f"Unexpected baseline order in {path.name}: {methods}")
        if np.atleast_1d(data["baseline_errors"]).shape != (5,):
            raise RuntimeError(f"Expected five baseline errors in {path.name}")
        truth = np.asarray(data["Rgt"]) @ np.asarray(data["XX"])
        for aligned, expected in zip(data["baseline_aligned"], data["baseline_errors"]):
            actual = np.sum((np.asarray(aligned) - truth) ** 2) / np.sum(truth ** 2)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        records["synthetic/" + path.name] = sha256(path)
    neural = Path(neural)
    for name in ["data.npz", "shared_projection.npz", "WA.npz", "DAD.npz", "HiWA-GMM.npz"]:
        path = neural / name
        if not path.is_file():
            raise RuntimeError(f"Missing neural input/baseline: {path}")
        records["neural/" + path.name] = sha256(path)
    with np.load(neural / "data.npz") as data:
        for name in ["source", "target", "source_labels", "target_labels"]:
            if name not in data:
                raise RuntimeError(f"data.npz missing {name}")
    return records


def validate_processed_data():
    config = json.loads((ROOT / "configs/experiments.json").read_text(encoding="utf-8"))
    data = ROOT / "data"
    records = []
    for groups, dimension in config["figure1"]["configs"]:
        for seed in range(config["figure1"]["seeds"]["start"],
                          config["figure1"]["seeds"]["stop"] + 1):
            path = data / "processed/synthetic" / f"S{groups:02d}_D{dimension:02d}_seed{seed:02d}.mat"
            item = loadmat(path, simplify_cells=True)
            source, target = np.asarray(item["XX"]), np.asarray(item["YY"])
            expected_shape = (dimension, groups * config["figure1"]["samples_per_group"])
            if int(item["S"]) != groups or source.shape != target.shape or source.shape != expected_shape:
                raise ValueError(f"Invalid synthetic input: {path}")
            records.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "sha256": sha256(path), "S": groups, "D": dimension,
                            "seed": seed, "shape": list(source.shape)})
    neural = []
    for name in ("neural_input.npz", "preprocessing.npz"):
        path = data / "processed/neural" / name
        with np.load(path) as item:
            neural.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                           "sha256": sha256(path),
                           "arrays": {key: list(item[key].shape) for key in item.files}})
    return {"schema_version": 2, "synthetic_count": len(records),
            "synthetic": records, "neural": neural}


def validate_results():
    missing = [relative for relative in REQUIRED_RESULTS if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))
    return {relative: {"bytes": (ROOT / relative).stat().st_size,
                       "sha256": sha256(ROOT / relative)} for relative in REQUIRED_RESULTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("all", "data", "author", "results"), default="all")
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()
    records = {}
    if args.scope in ("all", "data"):
        records["data"] = validate_processed_data()
    if args.scope in ("all", "author"):
        records["author"] = validate_author_inputs()
    if args.scope in ("all", "results"):
        records["results"] = validate_results()
    if args.write_manifest:
        args.write_manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote validation manifest: {args.write_manifest}")
    else:
        summary = {}
        if "data" in records:
            summary["processed_data"] = {
                "synthetic_files": records["data"]["synthetic_count"],
                "neural_files": len(records["data"]["neural"]),
            }
        if "author" in records:
            summary["author_reference_files"] = len(records["author"])
        if "results" in records:
            summary["required_results"] = len(records["results"])
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
