from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, write_json


def load_json(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def method_records(payload: dict, method: str) -> list[dict]:
    return sorted(
        (item for item in payload["results"] if item["method"] == method),
        key=lambda item: item["seed"],
    )


def summarize(records: list[dict]) -> dict:
    summary: dict[str, float | int | list[int]] = {
        "n_runs": len(records),
        "seeds": [int(item["seed"]) for item in records],
        "converged_runs": int(sum(bool(item["converged"]) for item in records)),
    }
    for key in ("after_direction_accuracy", "after_velocity_r2", "elapsed_seconds"):
        values = np.asarray([item[key] for item in records], dtype=float)
        summary[key] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def paired_summary(reference: list[dict], candidate: list[dict]) -> dict:
    reference_by_seed = {int(item["seed"]): item for item in reference}
    candidate_by_seed = {int(item["seed"]): item for item in candidate}
    seeds = sorted(set(reference_by_seed) & set(candidate_by_seed))
    result: dict[str, object] = {"seeds": seeds}
    rng = np.random.default_rng(20260705)
    for key in ("after_direction_accuracy", "after_velocity_r2"):
        differences = np.asarray(
            [candidate_by_seed[seed][key] - reference_by_seed[seed][key] for seed in seeds],
            dtype=float,
        )
        bootstrap = np.asarray(
            [rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(10000)]
        )
        result[key] = {
            "differences": differences.tolist(),
            "mean_difference": float(differences.mean()),
            "bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
    return result


def plot_temperature(development: dict[float, dict]) -> None:
    temperatures = np.asarray(sorted(development))
    accuracy_mean = []
    accuracy_std = []
    r2_mean = []
    r2_std = []
    for temperature in temperatures:
        records = method_records(development[float(temperature)], "prototype_soft")
        accuracy = np.asarray([item["after_direction_accuracy"] for item in records])
        r2 = np.asarray([item["after_velocity_r2"] for item in records])
        accuracy_mean.append(accuracy.mean())
        accuracy_std.append(accuracy.std())
        r2_mean.append(r2.mean())
        r2_std.append(r2.std())

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].errorbar(temperatures, accuracy_mean, yerr=accuracy_std, marker="o", capsize=5)
    axes[0].axhline(0.25, color="0.5", linestyle="--", linewidth=1, label="uniform 4-class chance")
    axes[0].set(title="Development seeds: direction accuracy", xlabel="softmax temperature", ylabel="accuracy")
    axes[0].set_ylim(0.2, 0.65)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].errorbar(temperatures, r2_mean, yerr=r2_std, marker="o", capsize=5, color="#b45309")
    axes[1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[1].set(title="Development seeds: movement score", xlabel="softmax temperature", ylabel="$R^2$")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(FIGURES_DIR / "soft_group_temperature_sensitivity.png", dpi=200)
    plt.close(figure)


def plot_heldout(random_payload: dict, warm_payload: dict) -> None:
    hard = method_records(warm_payload, "prototype_hard")
    random_soft = method_records(random_payload, "prototype_soft")
    warm_soft = method_records(warm_payload, "prototype_soft_warm")
    methods = (hard, random_soft, warm_soft)
    labels = ("prototype hard", "soft random init", "soft warm start")
    colors = ("#4a78a8", "#b54c4c", "#3c8c62")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, key, title, ylabel in (
        (axes[0], "after_direction_accuracy", "Held-out seeds: direction", "accuracy"),
        (axes[1], "after_velocity_r2", "Held-out seeds: movement", "$R^2$"),
    ):
        matrix = np.asarray([[item[key] for item in records] for records in methods]).T
        for row in matrix:
            axis.plot(range(3), row, color="0.78", linewidth=1, zorder=1)
        for index, (records, color) in enumerate(zip(methods, colors)):
            values = np.asarray([item[key] for item in records])
            axis.scatter(np.full(values.shape, index), values, color=color, s=35, zorder=3)
            axis.errorbar(
                index,
                values.mean(),
                yerr=values.std(),
                color="black",
                marker="D",
                markersize=5,
                capsize=4,
                zorder=4,
            )
        axis.set_xticks(range(3), labels, rotation=15, ha="right")
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0.3, 0.65)
    axes[1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    figure.savefig(FIGURES_DIR / "soft_group_heldout_comparison.png", dpi=200)
    plt.close(figure)


def plot_assignments() -> None:
    archive = np.load(RESULTS_DIR / "soft_neural_pilot_tau05_confirm_representative.npz")
    neural = archive["neural_3d"]
    movement = archive["movement_3d"]
    neural_assignments = archive["neural_assignments"]
    movement_assignments = archive["movement_assignments"]
    neural_prototypes = archive["neural_prototypes"]
    movement_prototypes = archive["movement_prototypes"]

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, values, assignments, prototypes, title in (
        (axes[0], neural, neural_assignments, neural_prototypes, "Neural soft groups (seed 5)"),
        (axes[1], movement, movement_assignments, movement_prototypes, "Movement soft groups (seed 5)"),
    ):
        group = np.argmax(assignments, axis=1)
        confidence = np.max(assignments, axis=1)
        axis.scatter(values[:, 0], values[:, 1], c=group, cmap="tab10", s=10, alpha=0.25 + 0.75 * confidence)
        axis.scatter(
            prototypes[:, 0],
            prototypes[:, 1],
            marker="X",
            s=120,
            c=np.arange(prototypes.shape[0]),
            cmap="tab10",
            edgecolors="black",
            linewidths=0.8,
        )
        axis.set(title=title, xlabel="dimension 1", ylabel="dimension 2")
        axis.grid(alpha=0.15)
    confidence = np.concatenate((np.max(neural_assignments, axis=1), np.max(movement_assignments, axis=1)))
    axes[2].hist(confidence, bins=np.linspace(0.25, 1.0, 16), color="#7551a6", alpha=0.85)
    axes[2].set(
        title="Maximum group membership",
        xlabel="largest assignment probability",
        ylabel="sample count",
    )
    axes[2].grid(axis="y", alpha=0.15)
    figure.savefig(FIGURES_DIR / "soft_group_assignment_structure.png", dpi=200)
    plt.close(figure)


def main() -> None:
    ensure_output_dirs()
    development = {
        1.0: load_json("soft_neural_pilot.json"),
        0.5: load_json("soft_neural_pilot_tau05.json"),
        0.25: load_json("soft_neural_pilot_tau025.json"),
    }
    random_confirm = load_json("soft_neural_pilot_tau05_confirm.json")
    warm_confirm = load_json("soft_neural_pilot_tau05_warm_confirm.json")
    hard = method_records(warm_confirm, "prototype_hard")
    random_soft = method_records(random_confirm, "prototype_soft")
    warm_soft = method_records(warm_confirm, "prototype_soft_warm")

    summary = {
        "scope": "exploratory five-seed development plus five held-out-seed confirmation",
        "development_temperature_sweep": {
            str(temperature): summarize(method_records(payload, "prototype_soft"))
            for temperature, payload in development.items()
        },
        "heldout_confirmation": {
            "prototype_hard": summarize(hard),
            "prototype_soft_random": summarize(random_soft),
            "prototype_soft_warm": summarize(warm_soft),
            "random_vs_hard_paired": paired_summary(hard, random_soft),
            "warm_vs_hard_paired": paired_summary(hard, warm_soft),
        },
        "decision": (
            "Random-init soft groups fail the go/no-go criterion on held-out seeds. "
            "Hard-solution warm start removes catastrophic failures but yields no mean direction-accuracy gain; "
            "retain as a stable implementation baseline, not as a demonstrated improvement."
        ),
    }
    write_json(RESULTS_DIR / "soft_group_experiment_summary.json", summary)
    plot_temperature(development)
    plot_heldout(random_confirm, warm_confirm)
    plot_assignments()
    print("saved soft-group summary and figures")


if __name__ == "__main__":
    main()
