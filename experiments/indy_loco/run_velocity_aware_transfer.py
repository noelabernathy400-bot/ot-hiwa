"""Continuous source-velocity supervision with target-unlabelled OT transfer."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RESULTS_DIR, ensure_output_dirs, write_json
from datasets.indy_loco import bin_indy_loco_session, load_indy_loco_session
from representation import DifferentiableVelocityAwareOT, MLPDecoder, MLPEncoder, VelocityAwareTransportConfig
from run_task_aware_transfer import _direction, _fit_transform, _indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=ROOT / "data" / "raw" / "indy_loco")
    parser.add_argument("--source-session", default="indy_20160915_01.mat")
    parser.add_argument("--target-session", default="indy_20160921_01.mat")
    parser.add_argument("--adaptation-fraction", type=float, default=0.60)
    parser.add_argument("--source-validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--input-dimension", type=int, default=8)
    parser.add_argument("--latent-dimension", type=int, default=8)
    parser.add_argument("--hidden-dimension", type=int, default=24)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--tag", default="indy_loco_velocity_aware_transfer_seed901_20260719")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.source_validation_fraction < args.adaptation_fraction < 1:
        raise ValueError("fractions must satisfy 0 < source validation < adaptation < 1")
    ensure_output_dirs(); np.random.seed(args.seed); torch.manual_seed(args.seed)
    source = bin_indy_loco_session(load_indy_loco_session(args.raw_directory / args.source_session))
    target = bin_indy_loco_session(load_indy_loco_session(args.raw_directory / args.target_session))
    train_end = int(len(source.neural_rates) * (args.adaptation_fraction - args.source_validation_fraction))
    validation_end = int(len(source.neural_rates) * args.adaptation_fraction)
    target_adaptation_end = int(len(target.neural_rates) * args.adaptation_fraction)
    train_i = _indices(train_end, args.max_samples)
    valid_i = _indices(validation_end - train_end, args.max_samples)
    adapt_i = _indices(target_adaptation_end, args.max_samples)
    test_i = _indices(len(target.neural_rates) - target_adaptation_end, args.max_samples)
    train_rates = source.neural_rates[:train_end][train_i]
    valid_rates = source.neural_rates[train_end:validation_end][valid_i]
    adapt_rates = target.neural_rates[:target_adaptation_end][adapt_i]
    test_rates = target.neural_rates[target_adaptation_end:][test_i]
    train_velocity = source.cursor_velocity[:train_end][train_i]
    valid_velocity = source.cursor_velocity[train_end:validation_end][valid_i]
    source_transform = _fit_transform(train_rates, args.input_dimension)
    target_transform = _fit_transform(adapt_rates, args.input_dimension)
    train = source_transform(train_rates).astype(np.float32)
    valid = source_transform(valid_rates).astype(np.float32)
    adaptation = target_transform(adapt_rates).astype(np.float32)
    target_test = target_transform(test_rates).astype(np.float32)
    velocity_scaler = StandardScaler().fit(train_velocity)
    scaled_train_velocity = velocity_scaler.transform(train_velocity).astype(np.float32)
    scaled_valid_velocity = velocity_scaler.transform(valid_velocity).astype(np.float32)
    groups = _direction(train_velocity, args.directions)
    model = DifferentiableVelocityAwareOT(
        MLPEncoder(args.input_dimension, args.latent_dimension, args.hidden_dimension),
        MLPEncoder(args.input_dimension, args.latent_dimension, args.hidden_dimension),
        MLPDecoder(args.latent_dimension, args.input_dimension, args.hidden_dimension),
        MLPDecoder(args.latent_dimension, args.input_dimension, args.hidden_dimension),
        nn.Linear(args.latent_dimension, 2), nn.Linear(args.latent_dimension, args.directions),
        n_groups=args.directions,
        config=VelocityAwareTransportConfig(source_warmup_epochs=args.warmup_epochs, epochs=args.epochs, sinkhorn_iterations=args.sinkhorn_iterations),
    )
    learned = model.fit(train, scaled_train_velocity, groups, adaptation, valid, scaled_valid_velocity)
    # This is the first access to target behaviour after all fitting/model selection.
    target_velocity = target.cursor_velocity[target_adaptation_end:][test_i]
    baseline_prediction = Ridge(alpha=1.0).fit(train, train_velocity).predict(target_test)
    learned_prediction = velocity_scaler.inverse_transform(model.predict_target_velocity(target_test))
    result = {
        "experiment": "indy_loco_continuous_velocity_source_supervised_target_unlabelled_ot",
        "scope": "fixed-seed qualification; target cursor position/velocity are absent from fitting",
        "label_usage": {"source_velocity": "source regression and source-only direction groups", "target_adaptation_velocity": "not passed to the model", "target_test_velocity": "opened only after fitting for final metrics"},
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "evaluation": {
            "source_only_velocity_r2": float(r2_score(target_velocity, baseline_prediction, multioutput="uniform_average")),
            "velocity_aware_ot_r2": float(r2_score(target_velocity, learned_prediction, multioutput="uniform_average")),
            "velocity_aware_ot_direction_accuracy": float(np.mean(_direction(learned_prediction, args.directions) == _direction(target_velocity, args.directions))),
            "source_validation_velocity_mse": float(learned.best_source_validation_mse),
        },
        "training": {"best_epoch": int(learned.best_epoch), "history": learned.history},
    }
    output = RESULTS_DIR / f"{args.tag}.json"; write_json(output, result)
    print(f"target velocity R2 source-only={result['evaluation']['source_only_velocity_r2']:.4f} velocity-aware={result['evaluation']['velocity_aware_ot_r2']:.4f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
