"""Task-aware, source-supervised / target-unlabelled Indy--Loco transfer.

Source cursor velocity is used only to define source direction labels and to
fit a source velocity decoder.  Target cursor data are opened only after the
model has been selected on a held-out source block.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RESULTS_DIR, ensure_output_dirs, write_json
from datasets.indy_loco import bin_indy_loco_session, load_indy_loco_session
from representation import DifferentiableTaskAwareOT, MLPDecoder, MLPEncoder, TaskAwareTransportConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=ROOT / "data" / "raw" / "indy_loco")
    parser.add_argument("--source-session", default="indy_20160915_01.mat")
    parser.add_argument("--target-session", default="indy_20160921_01.mat")
    parser.add_argument("--bin-width", type=float, default=0.05)
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
    parser.add_argument("--seed", type=int, default=801)
    parser.add_argument("--tag", default="indy_loco_task_aware_transfer_seed801_20260719")
    return parser.parse_args()


def _indices(n: int, maximum: int) -> np.ndarray:
    return np.arange(n) if n <= maximum else np.linspace(0, n - 1, maximum, dtype=int)


def _direction(velocity: np.ndarray, directions: int) -> np.ndarray:
    angle = np.arctan2(velocity[:, 1], velocity[:, 0])
    return np.floor(((angle + np.pi) % (2 * np.pi)) / (2 * np.pi) * directions).astype(np.int64)


def _fit_transform(train: np.ndarray, dimension: int):
    scaler = StandardScaler().fit(train)
    pca = PCA(n_components=dimension, random_state=0).fit(scaler.transform(train))
    latent_scaler = StandardScaler().fit(pca.transform(scaler.transform(train)))
    return lambda values: latent_scaler.transform(pca.transform(scaler.transform(values)))


def main() -> None:
    args = parse_args()
    if not 0 < args.adaptation_fraction < 1 or not 0 < args.source_validation_fraction < args.adaptation_fraction:
        raise ValueError("fractions must satisfy 0 < validation < adaptation < 1")
    ensure_output_dirs()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = bin_indy_loco_session(load_indy_loco_session(args.raw_directory / args.source_session), bin_width_seconds=args.bin_width)
    target = bin_indy_loco_session(load_indy_loco_session(args.raw_directory / args.target_session), bin_width_seconds=args.bin_width)

    source_train_end = int(len(source.neural_rates) * (args.adaptation_fraction - args.source_validation_fraction))
    source_validation_end = int(len(source.neural_rates) * args.adaptation_fraction)
    target_adaptation_end = int(len(target.neural_rates) * args.adaptation_fraction)
    source_train_index = _indices(source_train_end, args.max_samples)
    source_validation_index = _indices(source_validation_end - source_train_end, args.max_samples)
    target_adaptation_index = _indices(target_adaptation_end, args.max_samples)
    target_test_index = _indices(len(target.neural_rates) - target_adaptation_end, args.max_samples)

    source_train_rates = source.neural_rates[:source_train_end][source_train_index]
    source_validation_rates = source.neural_rates[source_train_end:source_validation_end][source_validation_index]
    target_adaptation_rates = target.neural_rates[:target_adaptation_end][target_adaptation_index]
    target_test_rates = target.neural_rates[target_adaptation_end:][target_test_index]
    source_train_velocity = source.cursor_velocity[:source_train_end][source_train_index]
    source_validation_velocity = source.cursor_velocity[source_train_end:source_validation_end][source_validation_index]

    source_transform = _fit_transform(source_train_rates, args.input_dimension)
    target_transform = _fit_transform(target_adaptation_rates, args.input_dimension)
    source_train = source_transform(source_train_rates).astype(np.float32)
    source_validation = source_transform(source_validation_rates).astype(np.float32)
    target_adaptation = target_transform(target_adaptation_rates).astype(np.float32)
    target_test = target_transform(target_test_rates).astype(np.float32)
    source_labels = _direction(source_train_velocity, args.directions)
    source_validation_labels = _direction(source_validation_velocity, args.directions)

    model = DifferentiableTaskAwareOT(
        MLPEncoder(args.input_dimension, args.latent_dimension, args.hidden_dimension),
        MLPEncoder(args.input_dimension, args.latent_dimension, args.hidden_dimension),
        MLPDecoder(args.latent_dimension, args.input_dimension, args.hidden_dimension),
        MLPDecoder(args.latent_dimension, args.input_dimension, args.hidden_dimension),
        nn.Linear(args.latent_dimension, args.directions),
        n_classes=args.directions,
        config=TaskAwareTransportConfig(
            source_warmup_epochs=args.warmup_epochs,
            epochs=args.epochs,
            sinkhorn_iterations=args.sinkhorn_iterations,
        ),
    )
    learned = model.fit(source_train, source_labels, target_adaptation, source_validation, source_validation_labels)

    # Target velocity and direction remain closed until after model.fit returns.
    target_test_velocity = target.cursor_velocity[target_adaptation_end:][target_test_index]
    target_test_labels = _direction(target_test_velocity, args.directions)
    baseline_decoder = Ridge(alpha=1.0).fit(source_train, source_train_velocity)
    learned_decoder = Ridge(alpha=1.0).fit(learned.source_latent, source_train_velocity)
    baseline_prediction = baseline_decoder.predict(target_test)
    learned_prediction = learned_decoder.predict(model.encode_target(target_test))
    target_direction = model.predict_target(target_test)
    payload = {
        "experiment": "indy_loco_task_aware_source_supervised_target_unlabelled_transfer",
        "scope": "single fixed-seed qualification run; source velocity creates source direction supervision, target behaviour is evaluation-only",
        "label_usage": {
            "source_velocity": "source direction labels and source ridge decoder only",
            "source_validation_velocity": "source-only checkpoint selection through direction accuracy",
            "target_adaptation_velocity": "not loaded into model fitting",
            "target_test_velocity": "opened after fitting only for R2 and direction accuracy",
        },
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "evaluation": {
            "source_only_velocity_r2": float(r2_score(target_test_velocity, baseline_prediction, multioutput="uniform_average")),
            "task_aware_velocity_r2": float(r2_score(target_test_velocity, learned_prediction, multioutput="uniform_average")),
            "task_aware_target_direction_accuracy": float(np.mean(target_direction == target_test_labels)),
            "source_validation_direction_accuracy": float(learned.best_source_validation_accuracy),
        },
        "training": {"best_epoch": int(learned.best_epoch), "history": learned.history},
    }
    output = RESULTS_DIR / f"{args.tag}.json"
    write_json(output, payload)
    print("target velocity R2 " f"source-only={payload['evaluation']['source_only_velocity_r2']:.4f} " f"task-aware={payload['evaluation']['task_aware_velocity_r2']:.4f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
