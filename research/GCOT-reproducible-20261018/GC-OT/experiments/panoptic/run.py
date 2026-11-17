"""Run the Panoptic 171204_pose2 experiment end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import run as compare
from .plot import run as plot
from .prepare import run as prepare


def run(raw_dir: Path, output_dir: Path) -> None:
    features = output_dir / "domains.npz"
    prepare(raw_dir, features)
    compare(features, output_dir)
    plot(output_dir / "aligned.npz", output_dir / "metrics.csv", output_dir)
    summary = {
        "status": "complete",
        "dataset": "CMU Panoptic 171204_pose2",
        "artifacts": {
            "diagnostic": "pca_tsne_three_groups.png",
            "comparison": "figure2a_panoptic.png",
            "metrics": "metrics.csv",
        },
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
