"""Cross-session neural decoder transfer on the public Indy--Loco data.

The primary question is practical: can an arm-velocity decoder trained on one
recording session be reused on a later session after *unlabelled* neural
alignment?  Target-session cursor position and velocity are not provided to
the alignment methods and are opened only after fitting for final metrics.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cc_hiwa.soft_groups import learn_soft_groups
from cc_hiwa.soft_hiwa import SoftHiWA
from cc_hiwa.common import RESULTS_DIR, ensure_output_dirs, write_json
from datasets.indy_loco import bin_indy_loco_session, load_indy_loco_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=ROOT / "data" / "raw" / "indy_loco")
    parser.add_argument("--source-session", default="indy_20160915_01.mat")
    parser.add_argument("--target-session", default="indy_20160921_01.mat")
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--min-firing-rate", type=float, default=0.5)
    parser.add_argument("--adaptation-fraction", type=float, default=0.60)
    parser.add_argument("--max-samples", type=int, default=192)
    parser.add_argument("--latent-dimension", type=int, default=8)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--consensus-weighting", choices=("uniform", "transport"), default="uniform")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--tag", default="indy_loco_cross_session_pilot_20160915_to_20160921_seed701")
    return parser.parse_args()


def _evenly_spaced(values: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or values.shape[0] <= maximum:
        return values
    indices = np.linspace(0, values.shape[0] - 1, maximum, dtype=int)
    return values[indices]


def _velocity_direction(velocity: np.ndarray, n_bins: int = 8) -> np.ndarray:
    angle = np.arctan2(velocity[:, 1], velocity[:, 0])
    return np.floor(((angle + np.pi) % (2.0 * np.pi)) / (2.0 * np.pi) * n_bins).astype(int)


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "velocity_r2": float(r2_score(truth, prediction, multioutput="uniform_average")),
        "direction_accuracy": float(np.mean(_velocity_direction(prediction) == _velocity_direction(truth))),
    }


def _prepare_domain(
    train_rates: np.ndarray,
    evaluation_rates: np.ndarray,
    *,
    latent_dimension: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    scaler = StandardScaler().fit(train_rates)
    train_scaled = scaler.transform(train_rates)
    evaluation_scaled = scaler.transform(evaluation_rates)
    pca = PCA(n_components=latent_dimension, random_state=0).fit(train_scaled)
    latent_scaler = StandardScaler().fit(pca.transform(train_scaled))
    train_latent = latent_scaler.transform(pca.transform(train_scaled))
    evaluation_latent = latent_scaler.transform(pca.transform(evaluation_scaled))
    return train_latent, evaluation_latent, {
        "input_dimension": int(train_rates.shape[1]),
        "latent_dimension": int(latent_dimension),
    }


def _fit_ridge(features: np.ndarray, velocity: np.ndarray, alpha: float) -> Ridge:
    return Ridge(alpha=alpha).fit(features, velocity)


def main() -> None:
    args = parse_args()
    if not 0 < args.adaptation_fraction < 1:
        raise ValueError("adaptation_fraction must lie strictly between zero and one")
    if args.max_samples < args.groups:
        raise ValueError("max_samples must be at least the number of groups")
    ensure_output_dirs()

    source = bin_indy_loco_session(
        load_indy_loco_session(args.raw_directory / args.source_session),
        bin_width_seconds=args.bin_width,
        min_firing_rate_hz=args.min_firing_rate,
    )
    target = bin_indy_loco_session(
        load_indy_loco_session(args.raw_directory / args.target_session),
        bin_width_seconds=args.bin_width,
        min_firing_rate_hz=args.min_firing_rate,
    )

    source_cut = int(source.neural_rates.shape[0] * args.adaptation_fraction)
    target_cut = int(target.neural_rates.shape[0] * args.adaptation_fraction)
    source_train_rates = _evenly_spaced(source.neural_rates[:source_cut], args.max_samples)
    source_train_velocity = _evenly_spaced(source.cursor_velocity[:source_cut], args.max_samples)
    target_adaptation_rates = _evenly_spaced(target.neural_rates[:target_cut], args.max_samples)
    target_test_rates = _evenly_spaced(target.neural_rates[target_cut:], args.max_samples)

    # Target-session velocity is deliberately withheld until after all fitting.
    target_test_velocity = _evenly_spaced(target.cursor_velocity[target_cut:], args.max_samples)

    source_latent, _, source_meta = _prepare_domain(
        source_train_rates, source_train_rates, latent_dimension=args.latent_dimension
    )
    target_adaptation_latent, target_test_latent, target_meta = _prepare_domain(
        target_adaptation_rates, target_test_rates, latent_dimension=args.latent_dimension
    )
    decoder = _fit_ridge(source_latent, source_train_velocity, args.ridge_alpha)

    source_groups = learn_soft_groups(
        source_latent,
        n_groups=args.groups,
        temperature=args.temperature,
        seed=args.seed,
    )
    target_groups = learn_soft_groups(
        target_adaptation_latent,
        n_groups=args.groups,
        temperature=args.temperature,
        seed=args.seed,
    )
    soft_model = SoftHiWA(
        dim_red_method=PCA(n_components=args.latent_dimension, random_state=args.seed),
        normalize=False,
        maxiter=args.maxiter,
        support_mode="full",
        random_state=args.seed,
        consensus_weighting=args.consensus_weighting,
    ).fit(
        target_adaptation_latent,
        target_groups.assignments,
        source_latent,
        source_groups.assignments,
    )
    hard_model = SoftHiWA(
        dim_red_method=PCA(n_components=args.latent_dimension, random_state=args.seed),
        normalize=False,
        maxiter=args.maxiter,
        support_mode="full",
        random_state=args.seed,
        consensus_weighting=args.consensus_weighting,
    ).fit(
        target_adaptation_latent,
        np.eye(args.groups)[np.argmax(target_groups.assignments, axis=1)],
        source_latent,
        np.eye(args.groups)[np.argmax(source_groups.assignments, axis=1)],
    )

    no_alignment = decoder.predict(target_test_latent)
    soft_prediction = decoder.predict(soft_model.transform(target_test_latent))
    hard_prediction = decoder.predict(hard_model.transform(target_test_latent))

    # Explicit oracle: target adaptation velocity trains a target-only decoder.
    # It is an upper-bound diagnostic and is never compared as an unsupervised method.
    target_oracle_velocity = _evenly_spaced(target.cursor_velocity[:target_cut], args.max_samples)
    oracle_prediction = _fit_ridge(target_adaptation_latent, target_oracle_velocity, args.ridge_alpha).predict(target_test_latent)

    payload = {
        "experiment": "indy_loco_cross_session_neural_decoder_transfer",
        "scope": "fixed first pilot; source session trains the decoder, target neural adaptation is unlabelled, and target velocity is evaluation-only except for the separately labelled oracle",
        "data_contract": {
            "source_session": args.source_session,
            "target_session": args.target_session,
            "source_velocity": "used to fit the source ridge decoder",
            "target_adaptation_neural_rates": "used without target cursor position or velocity for Soft-GCOT fitting",
            "target_test_neural_rates": "transformed after fitting only",
            "target_test_velocity": "opened only after all unsupervised fits for final metrics",
            "target_supervised_oracle": "separate labelled upper bound; not an alignment baseline",
        },
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": platform.python_version()},
        "binning": {
            "source": {"bins": int(source.neural_rates.shape[0]), "retained_units": int(source.neural_rates.shape[1])},
            "target": {"bins": int(target.neural_rates.shape[0]), "retained_units": int(target.neural_rates.shape[1])},
            "source_representation": source_meta,
            "target_representation": target_meta,
        },
        "evaluation": {
            "source_only_no_alignment": _metrics(no_alignment, target_test_velocity),
            "hard_group_hiwa": _metrics(hard_prediction, target_test_velocity),
            "soft_gcot": _metrics(soft_prediction, target_test_velocity),
            "target_supervised_oracle": _metrics(oracle_prediction, target_test_velocity),
        },
        "soft_gcot_diagnostics": soft_model.diagnostics,
        "hard_group_diagnostics": hard_model.diagnostics,
        "soft_group_diagnostics": {"source": source_groups.diagnostics, "target": target_groups.diagnostics},
    }
    output = RESULTS_DIR / f"{args.tag}.json"
    write_json(output, payload)
    print(f"target velocity R2: source-only={payload['evaluation']['source_only_no_alignment']['velocity_r2']:.4f} hard={payload['evaluation']['hard_group_hiwa']['velocity_r2']:.4f} soft={payload['evaluation']['soft_gcot']['velocity_r2']:.4f} oracle={payload['evaluation']['target_supervised_oracle']['velocity_r2']:.4f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
