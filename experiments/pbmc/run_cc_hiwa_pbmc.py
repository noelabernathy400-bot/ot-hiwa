"""Run CC-HiWA on prepared PBMC RNA-ATAC embeddings."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orthogonal_procrustes

HIWA_SCRIPTS = Path(__file__).resolve().parents[2] / "hiwa_python_reproduction" / "scripts"
if str(HIWA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(HIWA_SCRIPTS))

from cc_hiwa import coupling_entropy, fit_cc_hiwa  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5])
    parser.add_argument("--group-gamma", type=float, default=0.1)
    parser.add_argument("--sample-gamma", type=float, default=0.1)
    parser.add_argument("--sinkhorn-maxiter", type=int, default=300)
    parser.add_argument("--rotation-mode", choices=["none", "procrustes"], default="none")
    parser.add_argument("--tag", default="smoke")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _recall_at_k(score: np.ndarray, k: int) -> float:
    n = score.shape[0]
    if score.shape[0] != score.shape[1]:
        raise ValueError("paired retrieval requires equal source/target sample counts")
    k = min(k, n)
    topk = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
    hits = [index in topk[index] for index in range(n)]
    return float(np.mean(hits))


def _cell_type_transfer_accuracy(transport: np.ndarray, labels: np.ndarray | None) -> float | None:
    if labels is None:
        return None
    target_indices = np.argmax(transport, axis=1)
    predicted = labels[target_indices]
    return float(np.mean(predicted == labels))


def _row_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _similarity_retrieval_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _row_normalize(x) @ _row_normalize(y).T


def _procrustes_rotation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_x = x - x.mean(axis=0, keepdims=True)
    centered_y = y - y.mean(axis=0, keepdims=True)
    rotation, _ = orthogonal_procrustes(centered_x, centered_y)
    return rotation


def _baseline_metrics(x: np.ndarray, y: np.ndarray, labels: np.ndarray | None) -> dict[str, Any]:
    n = x.shape[0]
    raw_score = _similarity_retrieval_score(x, y)
    centered_x = x - x.mean(axis=0, keepdims=True)
    centered_y = y - y.mean(axis=0, keepdims=True)
    rotation = _procrustes_rotation(x, y)
    aligned_score = _similarity_retrieval_score(centered_x @ rotation, centered_y)
    return {
        "random_expectation": {
            "recall_at_1": 1.0 / n,
            "recall_at_5": min(5, n) / n,
            "recall_at_10": min(10, n) / n,
        },
        "raw_embedding_cosine": {
            "recall_at_1": _recall_at_k(raw_score, 1),
            "recall_at_5": _recall_at_k(raw_score, 5),
            "recall_at_10": _recall_at_k(raw_score, 10),
            "cell_type_transfer_accuracy": _cell_type_transfer_accuracy(raw_score, labels),
        },
        "orthogonal_procrustes_cosine": {
            "recall_at_1": _recall_at_k(aligned_score, 1),
            "recall_at_5": _recall_at_k(aligned_score, 5),
            "recall_at_10": _recall_at_k(aligned_score, 10),
            "cell_type_transfer_accuracy": _cell_type_transfer_accuracy(aligned_score, labels),
        },
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for record in records:
        key = str(record["beta"])
        output[key] = {
            "recall_at_1": record["recall_at_1"],
            "recall_at_5": record["recall_at_5"],
            "recall_at_10": record["recall_at_10"],
            "cell_type_transfer_accuracy": record["cell_type_transfer_accuracy"],
            "sample_transport_entropy": record["sample_transport_entropy"],
            "group_transport_entropy": record["group_transport_entropy"],
            "objective": record["objective"],
        }
    if "0.0" in output:
        baseline = output["0.0"]
        for value in output.values():
            value["delta_recall_at_1_vs_beta0"] = value["recall_at_1"] - baseline["recall_at_1"]
            value["delta_recall_at_5_vs_beta0"] = value["recall_at_5"] - baseline["recall_at_5"]
    return output


def _plot(summary: dict[str, Any], baselines: dict[str, Any], path: Path) -> None:
    betas = sorted(float(key) for key in summary)
    labels = [str(beta) for beta in betas]
    r1 = [summary[str(beta)]["recall_at_1"] for beta in betas]
    r5 = [summary[str(beta)]["recall_at_5"] for beta in betas]
    entropy = [summary[str(beta)]["sample_transport_entropy"] for beta in betas]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="white", constrained_layout=True)
    axes[0].plot(labels, r1, marker="o")
    axes[0].set(title="Paired retrieval Recall@1", xlabel="beta", ylabel="Recall@1")
    axes[0].axhline(baselines["random_expectation"]["recall_at_1"], color="gray", linestyle="--", linewidth=1, label="random")
    axes[0].axhline(baselines["raw_embedding_cosine"]["recall_at_1"], color="#ff7f0e", linestyle=":", linewidth=1.4, label="raw cosine")
    axes[0].axhline(baselines["orthogonal_procrustes_cosine"]["recall_at_1"], color="#d62728", linestyle="-.", linewidth=1.4, label="Procrustes")
    axes[1].plot(labels, r5, marker="o", color="#2ca02c")
    axes[1].set(title="Paired retrieval Recall@5", xlabel="beta", ylabel="Recall@5")
    axes[1].axhline(baselines["random_expectation"]["recall_at_5"], color="gray", linestyle="--", linewidth=1, label="random")
    axes[1].axhline(baselines["raw_embedding_cosine"]["recall_at_5"], color="#ff7f0e", linestyle=":", linewidth=1.4, label="raw cosine")
    axes[1].axhline(baselines["orthogonal_procrustes_cosine"]["recall_at_5"], color="#d62728", linestyle="-.", linewidth=1.4, label="Procrustes")
    axes[2].plot(labels, entropy, marker="o", color="#9467bd")
    axes[2].set(title="Sample transport entropy", xlabel="beta", ylabel="entropy")
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    x = data["rna_embedding"]
    y = data["atac_embedding"]
    a = data["rna_assignments"]
    b = data["atac_assignments"]
    labels = data["cell_type"] if "cell_type" in data.files else None
    baselines = _baseline_metrics(x, y, labels)
    rotation = None
    if args.rotation_mode == "procrustes":
        # fit_cc_hiwa applies x_aligned = (R @ x.T).T = x @ R.T.
        # scipy.linalg.orthogonal_procrustes returns Q for x @ Q ~= y, so pass Q.T.
        rotation = _procrustes_rotation(x, y).T
    records = []
    for beta in args.betas:
        result = fit_cc_hiwa(
            x,
            y,
            a,
            b,
            beta=beta,
            group_gamma=args.group_gamma,
            sample_gamma=args.sample_gamma,
            sinkhorn_maxiter=args.sinkhorn_maxiter,
            rotation=rotation,
        )
        score = result.sample_transport
        record = {
            "beta": beta,
            "recall_at_1": _recall_at_k(score, 1),
            "recall_at_5": _recall_at_k(score, 5),
            "recall_at_10": _recall_at_k(score, 10),
            "cell_type_transfer_accuracy": _cell_type_transfer_accuracy(score, labels),
            "sample_transport_entropy": coupling_entropy(result.sample_transport),
            "group_transport_entropy": coupling_entropy(result.group_transport),
            "objective": result.objective,
            "compatibility_mean": float(np.mean(result.compatibility)),
            "compatibility_min": float(np.min(result.compatibility)),
            "compatibility_max": float(np.max(result.compatibility)),
        }
        records.append(record)
        print(
            f"beta={beta:g} R@1={record['recall_at_1']:.4f} "
            f"R@5={record['recall_at_5']:.4f} entropy={record['sample_transport_entropy']:.4f}"
        )
    summary = _summarize(records)
    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = RESULTS_DIR / f"cc_hiwa_pbmc_raw{suffix}.json"
    summary_path = RESULTS_DIR / f"cc_hiwa_pbmc_summary{suffix}.json"
    figure_path = FIGURES_DIR / f"cc_hiwa_pbmc_summary{suffix}.png"
    payload = {
        "experiment": "cc_hiwa_pbmc",
        "input": str(args.input),
        "label_usage": "cell type labels are used only for optional final transfer evaluation",
        "parameters": {
            "betas": args.betas,
            "group_gamma": args.group_gamma,
            "sample_gamma": args.sample_gamma,
            "sinkhorn_maxiter": args.sinkhorn_maxiter,
            "rotation_mode": args.rotation_mode,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "baselines": baselines,
        "records": records,
    }
    _write_json(raw_path, payload)
    _write_json(summary_path, {"experiment": "cc_hiwa_pbmc", "baselines": baselines, "summary": summary})
    _plot(summary, baselines, figure_path)
    print(f"saved {raw_path}")
    print(f"saved {summary_path}")
    print(f"saved {figure_path}")


if __name__ == "__main__":
    main()
