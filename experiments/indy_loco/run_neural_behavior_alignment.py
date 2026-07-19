"""External HiWA-style neural-to-behaviour alignment on one Indy--Loco session.

Neural rates and cursor velocity are treated as unpaired distributions during
fitting.  Their shared timestamps are accessed only after fitting to evaluate
held-out neural-to-behaviour alignment.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RESULTS_DIR, ensure_output_dirs, write_json
from datasets.indy_loco import bin_indy_loco_session, load_indy_loco_session
from global_soft_gcot import GlobalSoftGCOT
from soft_groups import learn_soft_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=ROOT / "data" / "raw" / "indy_loco")
    parser.add_argument("--session", default="indy_20160915_01.mat")
    parser.add_argument("--adaptation-fraction", type=float, default=0.60)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--sinkhorn-maxiter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--tag", default="indy_loco_neural_behavior_global_soft_gcot_seed1001_20260719")
    return parser.parse_args()


def _indices(n: int, maximum: int) -> np.ndarray:
    return np.arange(n) if n <= maximum else np.linspace(0, n - 1, maximum, dtype=int)


def _direction(values: np.ndarray) -> np.ndarray:
    return np.floor(((np.arctan2(values[:, 1], values[:, 0]) + np.pi) % (2 * np.pi)) / (2 * np.pi) * 8).astype(int)


def _fit_pca(values: np.ndarray, dimension: int):
    scaler = StandardScaler().fit(values)
    pca = PCA(n_components=dimension, random_state=0).fit(scaler.transform(values))
    output_scaler = StandardScaler().fit(pca.transform(scaler.transform(values)))
    return lambda data: output_scaler.transform(pca.transform(scaler.transform(data)))


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "velocity_r2": float(r2_score(truth, prediction, multioutput="uniform_average")),
        "direction_accuracy": float(np.mean(_direction(prediction) == _direction(truth))),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.adaptation_fraction < 1 or min(args.max_samples, args.groups, args.dimension) <= 0:
        raise ValueError("invalid fractions, sample count, group count, or dimension")
    ensure_output_dirs()
    session = bin_indy_loco_session(load_indy_loco_session(args.raw_directory / args.session))
    split = int(len(session.neural_rates) * args.adaptation_fraction)
    adaptation_i = _indices(split, args.max_samples)
    test_i = _indices(len(session.neural_rates) - split, args.max_samples)
    neural_adaptation = session.neural_rates[:split][adaptation_i]
    behavior_adaptation = session.cursor_velocity[:split][adaptation_i]
    neural_test = session.neural_rates[split:][test_i]
    # Behaviour pairs are withheld until after both OT fits below.
    behavior_test = session.cursor_velocity[split:][test_i]
    neural_transform = _fit_pca(neural_adaptation, args.dimension)
    behavior_scaler = StandardScaler().fit(behavior_adaptation)
    neural_adaptation_z = neural_transform(neural_adaptation)
    behavior_adaptation_z = behavior_scaler.transform(behavior_adaptation)
    neural_test_z = neural_transform(neural_test)
    neural_groups = learn_soft_groups(neural_adaptation_z, args.groups, args.temperature, seed=args.seed)
    behavior_groups = learn_soft_groups(behavior_adaptation_z, args.groups, args.temperature, seed=args.seed)
    settings = dict(maxiter=args.maxiter, sinkhorn_maxiter=args.sinkhorn_maxiter)
    soft = GlobalSoftGCOT(**settings).fit(neural_adaptation_z, neural_groups.assignments, behavior_adaptation_z, behavior_groups.assignments)
    hard = GlobalSoftGCOT(**settings).fit(
        neural_adaptation_z, np.eye(args.groups)[np.argmax(neural_groups.assignments, axis=1)],
        behavior_adaptation_z, np.eye(args.groups)[np.argmax(behavior_groups.assignments, axis=1)],
    )
    # Evaluation-only paired oracle; not used in groups, OT, or selection.
    u, _, vh = np.linalg.svd(behavior_adaptation_z.T @ neural_adaptation_z)
    oracle_rotation = u @ vh
    no_alignment = behavior_scaler.inverse_transform(neural_test_z)
    soft_prediction = behavior_scaler.inverse_transform(soft.transform(neural_test_z))
    hard_prediction = behavior_scaler.inverse_transform(hard.transform(neural_test_z))
    oracle_prediction = behavior_scaler.inverse_transform((oracle_rotation @ neural_test_z.T).T)
    result = {
        "experiment": "indy_loco_heldout_neural_to_behavior_distribution_alignment",
        "scope": "external HiWA-style evaluation; timestamps are withheld from group/OT fitting and used only for held-out metrics plus the explicit paired oracle",
        "data_contract": {"neural_rates": "source distribution", "cursor_velocity": "target distribution", "adaptation_pairs": "not provided to OT", "test_pairs": "opened only for final evaluation", "paired_oracle": "evaluation-only upper-bound diagnostic"},
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": platform.python_version()},
        "evaluation": {"no_alignment": _metrics(no_alignment, behavior_test), "hard_global_hiwa": _metrics(hard_prediction, behavior_test), "soft_global_gcot": _metrics(soft_prediction, behavior_test), "paired_procrustes_oracle": _metrics(oracle_prediction, behavior_test)},
        "diagnostics": {"hard": hard.diagnostics, "soft": soft.diagnostics},
    }
    output = RESULTS_DIR / f"{args.tag}.json"; write_json(output, result)
    print(f"held-out velocity R2 no-align={result['evaluation']['no_alignment']['velocity_r2']:.4f} hard={result['evaluation']['hard_global_hiwa']['velocity_r2']:.4f} soft={result['evaluation']['soft_global_gcot']['velocity_r2']:.4f} oracle={result['evaluation']['paired_procrustes_oracle']['velocity_r2']:.4f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
