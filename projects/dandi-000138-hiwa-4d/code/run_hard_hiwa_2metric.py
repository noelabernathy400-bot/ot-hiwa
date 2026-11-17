"""Run original archived Hard HiWA, then publish only rMSE-proxy and standard R² as primary evaluation.

This wrapper does not alter hiwa_original.py, and keeps old NRMSE in the
legacy run JSON solely as historical reproducibility evidence.
"""
import subprocess
import sys
from pathlib import Path


def main():
    # Forward all original runner CLI arguments verbatim.
    args = sys.argv[1:]
    root = Path(__file__).resolve().parent
    cmd = [sys.executable, str(root / "run_hard_hiwa.py"), *args]
    subprocess.run(cmd, check=True)
    # Take the exact input and output paths from the original runner flags.
    def value(flag):
        i = args.index(flag)
        if i + 1 >= len(args):
            raise ValueError(f"Missing value for {flag}")
        return args[i+1]
    clusters = value("--clusters")
    output = Path(value("--output"))
    subprocess.run([
        sys.executable, str(root / "evaluate_hiwa_two_metrics.py"),
        "--clusters", clusters,
        "--alignment", str(output / "hard_hiwa_results.json"),
        "--output", str(output / "hiwa_rmse_r2_evaluation.json")
    ], check=True)


if __name__ == "__main__":
    main()
