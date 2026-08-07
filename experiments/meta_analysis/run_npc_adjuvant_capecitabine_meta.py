"""Reproduce the preliminary trial-level meta-analysis in the NPC note.

The two independent RCTs and their published hazard ratios are encoded below.
Chen 2021 and Chen 2026 are reports from the same randomized cohort: use the
2021 report for FFS and the 2026 long-term report for OS, never as two studies
within the same outcome analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class HazardRatio:
    label: str
    hazard_ratio: float
    lower_ci: float
    upper_ci: float

    @property
    def log_hr(self) -> float:
        return math.log(self.hazard_ratio)

    @property
    def standard_error(self) -> float:
        return (math.log(self.upper_ci) - math.log(self.lower_ci)) / (2.0 * 1.96)


FFS_STUDIES = (
    HazardRatio("Chen 2021 (metronomic)", 0.50, 0.32, 0.79),
    HazardRatio("Miao 2022 (eight cycles)", 0.53, 0.30, 0.94),
)

OS_STUDIES = (
    HazardRatio("Chen 2026 long-term (metronomic)", 0.53, 0.31, 0.91),
    HazardRatio("Miao 2022 (eight cycles)", 0.62, 0.29, 1.32),
)


def fixed_effect_meta(studies: tuple[HazardRatio, ...]) -> dict[str, float | list[float]]:
    """Pool log hazard ratios by inverse-variance fixed effects."""
    log_effects = [study.log_hr for study in studies]
    standard_errors = [study.standard_error for study in studies]
    weights = [1.0 / standard_error**2 for standard_error in standard_errors]
    pooled_log_hr = sum(weight * value for weight, value in zip(weights, log_effects)) / sum(weights)
    pooled_se = math.sqrt(1.0 / sum(weights))
    q_statistic = sum(
        weight * (value - pooled_log_hr) ** 2
        for weight, value in zip(weights, log_effects)
    )
    degrees_of_freedom = len(studies) - 1
    i_squared = 0.0 if q_statistic == 0 else max(0.0, (q_statistic - degrees_of_freedom) / q_statistic) * 100.0
    return {
        "pooled_hr": math.exp(pooled_log_hr),
        "lower_ci": math.exp(pooled_log_hr - 1.96 * pooled_se),
        "upper_ci": math.exp(pooled_log_hr + 1.96 * pooled_se),
        "pooled_log_hr": pooled_log_hr,
        "pooled_se": pooled_se,
        "q": q_statistic,
        "i_squared_percent": i_squared,
        "weights_percent": [100.0 * weight / sum(weights) for weight in weights],
    }


def _draw_panel(axis: plt.Axes, title: str, studies: tuple[HazardRatio, ...], summary: dict[str, float | list[float]]) -> None:
    positions = list(range(len(studies), 0, -1))
    for position, study in zip(positions, studies):
        axis.errorbar(
            study.hazard_ratio,
            position,
            xerr=[[study.hazard_ratio - study.lower_ci], [study.upper_ci - study.hazard_ratio]],
            fmt="s",
            color="#1f4e79",
            markersize=7,
            capsize=3,
        )
    pooled_position = 0
    pooled_hr = float(summary["pooled_hr"])
    pooled_lower = float(summary["lower_ci"])
    pooled_upper = float(summary["upper_ci"])
    axis.errorbar(
        pooled_hr,
        pooled_position,
        xerr=[[pooled_hr - pooled_lower], [pooled_upper - pooled_hr]],
        fmt="D",
        color="#8b1e3f",
        markersize=8,
        capsize=3,
    )
    axis.axvline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xscale("log")
    axis.set_xlim(0.2, 1.6)
    axis.set_xticks([0.25, 0.5, 1.0, 1.5])
    axis.set_xticklabels(["0.25", "0.50", "1.00", "1.50"])
    axis.set_yticks(positions + [pooled_position])
    axis.set_yticklabels([study.label for study in studies] + ["Fixed-effect pooled estimate"])
    axis.set_ylim(-0.6, len(studies) + 0.7)
    axis.set_xlabel("Hazard ratio (capecitabine vs observation)")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.text(0.205, -0.45, "Favours capecitabine", ha="left", va="center", fontsize=9)
    axis.text(1.57, -0.45, "Favours observation", ha="right", va="center", fontsize=9)
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.7)


def make_forest_plot(output_path: Path, ffs: dict[str, float | list[float]], os: dict[str, float | list[float]]) -> None:
    """Create a compact forest plot for the two prespecified time-to-event outcomes."""
    figure, axes = plt.subplots(2, 1, figsize=(9, 6.8), constrained_layout=True)
    _draw_panel(axes[0], "A. Failure-free survival", FFS_STUDIES, ffs)
    _draw_panel(axes[1], "B. Overall survival", OS_STUDIES, os)
    figure.suptitle("Adjuvant capecitabine after chemoradiotherapy in high-risk LA-NPC", fontsize=13, fontweight="bold")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/research/figures/npc_adjuvant_capecitabine_meta"),
        help="Directory for the derived forest plot and numerical summary.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ffs = fixed_effect_meta(FFS_STUDIES)
    os = fixed_effect_meta(OS_STUDIES)
    payload = {"failure_free_survival": ffs, "overall_survival": os}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_forest_plot(args.output_dir / "forest_plot.png", ffs, os)

    for outcome, summary in payload.items():
        print(
            f"{outcome}: HR={summary['pooled_hr']:.4f} "
            f"(95% CI {summary['lower_ci']:.4f}-{summary['upper_ci']:.4f}), "
            f"Q={summary['q']:.4f}, I2={summary['i_squared_percent']:.1f}%"
        )


if __name__ == "__main__":
    main()
