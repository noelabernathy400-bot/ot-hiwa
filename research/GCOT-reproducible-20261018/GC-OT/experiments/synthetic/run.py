"""Run the synthetic Figure 1 experiments without touching neural outputs."""
import argparse

from experiments.synthetic.core import run_synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.trials <= 50:
        parser.error("--trials must be in [1, 50]")
    run_synthetic(args.trials, args.workers)
    if args.render:
        from visualization.synthetic import main as render
        render()


if __name__ == "__main__":
    main()
