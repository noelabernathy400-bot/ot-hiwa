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
from cc_hiwa.global_soft_gcot import GlobalSoftGCOT
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
    parser.add_argument(
        "--source-grouping",
        choices=("neural", "velocity"),
        default="neural",
        help=(
            "How source soft groups are defined. 'velocity' is source-supervised "
            "and is only permitted because source cursor velocity trains the transfer decoder."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument(
        "--global-sinkhorn-maxiter",
        type=int,
        default=100,
        help="Inner log-Sinkhorn iterations when alignment-solver=global_rotation.",
    )
    parser.add_argument(
        "--alignment-solver",
        choices=("local_admm", "global_rotation"),
        default="local_admm",
        help=(
            "'global_rotation' keeps hierarchical OT but replaces local rotations plus ADMM "
            "with one shared orthogonal map."
        ),
    )
    parser.add_argument("--consensus-weighting", choices=("uniform", "transport"), default="uniform")
    parser.add_argument(
        "--temporal-signature-weight",
        type=float,
        default=0.0,
        help=(
            "Optional unlabelled temporal-dynamics regulariser for local OT. "
            "Zero preserves the frozen baseline."
        ),
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--tag", default="indy_loco_cross_session_pilot_20160915_to_20160921_seed701")
    return parser.parse_args()


def _evenly_spaced_indices(n_samples: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or n_samples <= maximum:
        return np.arange(n_samples)
    return np.linspace(0, n_samples - 1, maximum, dtype=int)


def _velocity_direction(velocity: np.ndarray, n_bins: int = 8) -> np.ndarray:
    angle = np.arctan2(velocity[:, 1], velocity[:, 0])
    return np.floor(((angle + np.pi) % (2.0 * np.pi)) / (2.0 * np.pi) * n_bins).astype(int)


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "velocity_r2": float(r2_score(truth, prediction, multioutput="uniform_average")),
        "direction_accuracy": float(np.mean(_velocity_direction(prediction) == _velocity_direction(truth))),
    }


def _fit_domain_transform(train_rates: np.ndarray, *, latent_dimension: int):
    scaler = StandardScaler().fit(train_rates)
    train_scaled = scaler.transform(train_rates)
    pca = PCA(n_components=latent_dimension, random_state=0).fit(train_scaled)
    latent_scaler = StandardScaler().fit(pca.transform(train_scaled))
    def transform(rates: np.ndarray) -> np.ndarray:
        return latent_scaler.transform(pca.transform(scaler.transform(rates)))
    return transform, {
        "input_dimension": int(train_rates.shape[1]),
        "latent_dimension": int(latent_dimension),
    }


def _temporal_signatures(latent: np.ndarray, lags: tuple[int, ...] = (1, 2, 4, 8)) -> np.ndarray:
    """Causal, unlabelled local dynamics signatures for matching neural states.

    Each feature records the log displacement over a past lag.  The signature
    deliberately uses neural latents only: cursor coordinates and velocities
    never enter the OT solve.
    """
    values = np.asarray(latent, dtype=float)
    features = []
    for lag in lags:
        previous = np.vstack((np.repeat(values[:1], lag, axis=0), values[:-lag]))
        features.append(np.log1p(np.linalg.norm(values - previous, axis=1)))
    return StandardScaler().fit_transform(np.column_stack(features))


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
    source_indices = _evenly_spaced_indices(source_cut, args.max_samples)
    target_adaptation_indices = _evenly_spaced_indices(target_cut, args.max_samples)
    target_test_indices = _evenly_spaced_indices(target.neural_rates.shape[0] - target_cut, args.max_samples)
    source_train_rates = source.neural_rates[:source_cut][source_indices]
    source_train_velocity = source.cursor_velocity[:source_cut][source_indices]
    target_adaptation_rates = target.neural_rates[:target_cut][target_adaptation_indices]
    target_test_rates = target.neural_rates[target_cut:][target_test_indices]

    # Target-session velocity is deliberately withheld until after all fitting.
    target_test_velocity = target.cursor_velocity[target_cut:][target_test_indices]

    source_transform, source_meta = _fit_domain_transform(
        source_train_rates, latent_dimension=args.latent_dimension
    )
    target_transform, target_meta = _fit_domain_transform(
        target_adaptation_rates, latent_dimension=args.latent_dimension
    )
    source_latent = source_transform(source_train_rates)
    target_adaptation_latent = target_transform(target_adaptation_rates)
    target_test_latent = target_transform(target_test_rates)
    source_temporal_signatures = _temporal_signatures(
        source_transform(source.neural_rates[:source_cut])
    )[source_indices]
    target_temporal_signatures = _temporal_signatures(
        target_transform(target.neural_rates[:target_cut])
    )[target_adaptation_indices]
    decoder = _fit_ridge(source_latent, source_train_velocity, args.ridge_alpha)

    source_group_values = source_latent if args.source_grouping == "neural" else source_train_velocity
    source_groups = learn_soft_groups(
        source_group_values,
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
    if args.alignment_solver == "global_rotation":
        if args.temporal_signature_weight != 0:
            raise ValueError("temporal signatures are currently implemented only for local_admm")
        solver_settings = dict(
            maxiter=args.maxiter,
            sinkhorn_maxiter=args.global_sinkhorn_maxiter,
        )
        soft_model = GlobalSoftGCOT(**solver_settings).fit(
            target_adaptation_latent,
            target_groups.assignments,
            source_latent,
            source_groups.assignments,
        )
        hard_model = GlobalSoftGCOT(**solver_settings).fit(
            target_adaptation_latent,
            np.eye(args.groups)[np.argmax(target_groups.assignments, axis=1)],
            source_latent,
            np.eye(args.groups)[np.argmax(source_groups.assignments, axis=1)],
        )
    else:
        solver_settings = dict(
            dim_red_method=PCA(n_components=args.latent_dimension, random_state=args.seed),
            normalize=False,
            maxiter=args.maxiter,
            support_mode="full",
            random_state=args.seed,
            consensus_weighting=args.consensus_weighting,
            temporal_signature_weight=args.temporal_signature_weight,
        )
        soft_model = SoftHiWA(**solver_settings).fit(
            target_adaptation_latent,
            target_groups.assignments,
            source_latent,
            source_groups.assignments,
            source_temporal_signatures=target_temporal_signatures,
            target_temporal_signatures=source_temporal_signatures,
        )
        hard_model = SoftHiWA(**solver_settings).fit(
            target_adaptation_latent,
            np.eye(args.groups)[np.argmax(target_groups.assignments, axis=1)],
            source_latent,
            np.eye(args.groups)[np.argmax(source_groups.assignments, axis=1)],
            source_temporal_signatures=target_temporal_signatures,
            target_temporal_signatures=source_temporal_signatures,
        )

    no_alignment = decoder.predict(target_test_latent)
    soft_prediction = decoder.predict(soft_model.transform(target_test_latent))
    hard_prediction = decoder.predict(hard_model.transform(target_test_latent))

    # Explicit oracle: target adaptation velocity trains a target-only decoder.
    # It is an upper-bound diagnostic and is never compared as an unsupervised method.
    target_oracle_velocity = target.cursor_velocity[:target_cut][target_adaptation_indices]
    oracle_prediction = _fit_ridge(target_adaptation_latent, target_oracle_velocity, args.ridge_alpha).predict(target_test_latent)

    payload = {
        "experiment": "indy_loco_cross_session_neural_decoder_transfer",
        "scope": "fixed first pilot; source session trains the decoder, target neural adaptation is unlabelled, and target velocity is evaluation-only except for the separately labelled oracle",
        "data_contract": {
            "source_session": args.source_session,
            "target_session": args.target_session,
            "source_velocity": "used to fit the source ridge decoder",
            "source_grouping": (
                "source neural latent states"
                if args.source_grouping == "neural"
                else "source cursor velocity only; source-supervised state anchor"
            ),
            "target_adaptation_neural_rates": "used without target cursor position or velocity for Soft-GCOT fitting",
            "target_test_neural_rates": "transformed after fitting only",
            "target_test_velocity": "opened only after all unsupervised fits for final metrics",
            "target_supervised_oracle": "separate labelled upper bound; not an alignment baseline",
            "temporal_signature": (
                "causal neural-latent displacement signature, used without target behaviour"
                if args.temporal_signature_weight > 0
                else "disabled"
            ),
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
