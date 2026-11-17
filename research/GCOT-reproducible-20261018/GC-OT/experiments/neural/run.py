"""Run the neural GCOT fit and Figure 2a."""
import argparse

from experiments.neural.core import fit, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "render", "all"), default="all")
    parser.add_argument("--skip-core-fit", action="store_true")
    args = parser.parse_args()
    if args.stage in ("fit", "all"):
        if not args.skip_core_fit:
            fit()
    if args.stage in ("render", "all"):
        render()
        from visualization.neural import main as render_comparison
        render_comparison()


if __name__ == "__main__":
    main()
