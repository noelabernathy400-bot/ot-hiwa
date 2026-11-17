"""Run the accepted DANDI 000688 experiment from raw NWB files to figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import run as compare
from .config import PREPROCESSING, SOURCE_FILENAME, TARGET_FILENAME
from .plot import run as plot
from .prepare import run as prepare


def run(raw_dir: Path, output_dir: Path) -> None:
    source = raw_dir / SOURCE_FILENAME
    target = raw_dir / TARGET_FILENAME
    latents = output_dir / "latents.npz"
    aligned = output_dir / "aligned.npz"
    metrics = output_dir / "metrics.csv"

    prepare(source, target, latents, dimension=PREPROCESSING.latent_dimension)
    compare(latents, output_dir)
    plot(aligned, metrics, output_dir)

    summary = {
        "status": "complete",
        "source": source.name,
        "target": target.name,
        "artifacts": {
            "latents": latents.name,
            "aligned": aligned.name,
            "metrics": metrics.name,
            "figure2a": "figure2a_dandi.png",
            "figure2a_vector": "figure2a_dandi.svg",
            "figure2a_pdf": "figure2a_dandi.pdf",
            "accuracy_plot": "alignment_accuracy.png",
            "accuracy_plot_vector": "alignment_accuracy.pdf",
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
