from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from diagnose_seed7 import _direction_purity, _nn_error_analysis  # noqa: E402
from analyze_roca_degeneracy_diagnostics import (  # noqa: E402
    _matching_diagnostics,
    _simplex_diagnostics,
    _warning_v0,
)
from calibrate_roca_confidence_thresholds import _warning_with_thresholds  # noqa: E402
from calibrate_roca_confidence import (  # noqa: E402
    _evaluate_rule,
    _metric_warning,
    split_records,
)
from run_component_aware import _representative_orientation_selector  # noqa: E402
from run_roca_synthetic_determinant_validation import run_case  # noqa: E402


class DiagnosticLabelTests(unittest.TestCase):
    def test_direction_purity_preserves_original_label_values(self) -> None:
        assignments = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        labels = np.asarray([3, 3, 7, 7])
        purity, classes = _direction_purity(assignments, labels)
        np.testing.assert_array_equal(classes, [3, 7])
        np.testing.assert_allclose(purity, np.eye(2))

    def test_nn_error_analysis_reports_original_labels(self) -> None:
        true = np.asarray([3, 3, 4, 4, 5, 5, 7, 7])
        predicted = np.asarray([4, 4, 3, 4, 4, 7, 7, 7])
        result = _nn_error_analysis(true, predicted)
        self.assertEqual(result["class_labels"], [3, 4, 5, 7])
        self.assertEqual(result["top_errors"][0]["true_direction"], 3)
        self.assertEqual(result["top_errors"][0]["predicted_as"], 4)
        self.assertEqual(result["per_class_accuracy"]["5"]["most_confused_with"], 4)

    def test_representative_orientation_detects_reflection_component(self) -> None:
        source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        reflection = np.diag([-1.0, 1.0, 1.0])
        target = source @ reflection.T
        assignments = np.eye(4)
        result = _representative_orientation_selector(
            source,
            assignments,
            target,
            assignments,
            [np.eye(4) / 4.0, np.eye(4) / 4.0],
        )
        self.assertEqual(result["representative_orientation_sign"], -1)
        self.assertGreater(abs(result["representative_orientation_product"]), 1e-6)

    def test_representative_orientation_flips_when_source_axis_is_reflected(self) -> None:
        source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        reflection = np.diag([-1.0, 1.0, 1.0])
        target = source @ reflection.T
        assignments = np.eye(4)
        baseline = _representative_orientation_selector(
            source,
            assignments,
            target,
            assignments,
            [np.eye(4) / 4.0, np.eye(4) / 4.0],
        )
        source_reflected = source.copy()
        source_reflected[:, 0] *= -1.0
        flipped = _representative_orientation_selector(
            source_reflected,
            assignments,
            target,
            assignments,
            [np.eye(4) / 4.0, np.eye(4) / 4.0],
        )
        self.assertEqual(flipped["representative_orientation_sign"], -baseline["representative_orientation_sign"])

    def test_synthetic_roca_selector_recovers_noiseless_true_determinants(self) -> None:
        args = argparse.Namespace(
            points_per_cluster=20,
            cluster_std=0.04,
            oracle_confidence=0.97,
            learned_temperature=0.50,
            learned_entropy_weight=0.05,
            degeneracy_epsilon=1e-5,
        )
        for true_det in (-1, 1):
            result = run_case(
                seed=13,
                true_det=true_det,
                noise_level=0.0,
                assignment_mode="oracle_soft",
                degeneracy_mode="normal",
                args=args,
            )
            self.assertEqual(result["selected_det"], true_det)
            self.assertTrue(result["selected_is_correct"])
            self.assertIsNone(result["selector_error"])

    def test_degeneracy_diagnostics_flag_ill_conditioned_simplex(self) -> None:
        representatives = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1e-4],
            ]
        )
        diagnostics = _simplex_diagnostics(representatives)
        self.assertLess(diagnostics["abs_volume"], 1e-3)
        self.assertGreater(diagnostics["condition_number"], 1e3)

    def test_matching_diagnostics_detects_ambiguous_assignment(self) -> None:
        transport = np.asarray(
            [
                [0.26, 0.24, 0.00, 0.00],
                [0.24, 0.26, 0.00, 0.00],
                [0.00, 0.00, 0.25, 0.00],
                [0.00, 0.00, 0.00, 0.25],
            ]
        )
        diagnostics = _matching_diagnostics(transport)
        self.assertLess(diagnostics["matching_score_relative_margin"], 0.05)

    def test_warning_v0_is_diagnostic_not_label_based(self) -> None:
        warning = _warning_v0(
            {
                "volume_product_margin": 0.5,
                "source_condition_number": 100.0,
                "target_condition_number": 10.0,
                "source_assignment_mean_entropy": 0.5,
                "target_assignment_mean_entropy": 0.2,
                "matching_score_relative_margin": 0.01,
            }
        )
        self.assertTrue(warning["warning_v0"])
        self.assertIn("low_oriented_volume_margin", warning["warning_v0_reasons"])
        self.assertIn("ill_conditioned_simplex", warning["warning_v0_reasons"])
        self.assertIn("high_assignment_entropy", warning["warning_v0_reasons"])
        self.assertIn("ambiguous_group_matching", warning["warning_v0_reasons"])

    def test_calibrated_warning_uses_fixed_unlabeled_thresholds(self) -> None:
        thresholds = {
            "volume_product_margin_min": 0.0,
            "simplex_condition_number_max": 80.0,
            "assignment_entropy_max": 1e12,
            "matching_relative_margin_min": 0.0,
        }
        safe = _warning_with_thresholds(
            {
                "volume_product_margin": 0.5,
                "source_condition_number": 30.0,
                "target_condition_number": 20.0,
                "source_assignment_mean_entropy": 0.9,
                "target_assignment_mean_entropy": 0.9,
                "matching_score_relative_margin": 0.001,
            },
            thresholds,
        )
        self.assertFalse(safe["warning"])
        flagged = _warning_with_thresholds(
            {
                "volume_product_margin": 50.0,
                "source_condition_number": 81.0,
                "target_condition_number": 10.0,
                "source_assignment_mean_entropy": 0.1,
                "target_assignment_mean_entropy": 0.1,
                "matching_score_relative_margin": 0.5,
            },
            thresholds,
        )
        self.assertTrue(flagged["warning"])
        self.assertEqual(flagged["warning_reasons"], ["ill_conditioned_simplex"])

    def test_confidence_split_rejects_overlapping_seeds(self) -> None:
        records = [{"seed": 0}, {"seed": 1}]
        with self.assertRaises(ValueError):
            split_records(records, {0, 1}, {1})

    def test_confidence_split_is_disjoint(self) -> None:
        records = [{"seed": seed} for seed in range(4)]
        calibration, validation = split_records(records, {0, 1}, {2, 3})
        self.assertEqual({record["seed"] for record in calibration}, {0, 1})
        self.assertEqual({record["seed"] for record in validation}, {2, 3})

    def test_metric_warning_outputs_bool(self) -> None:
        self.assertIsInstance(
            _metric_warning({"volume_product_margin": 0.1}, "volume_product_margin", "low", 0.5),
            bool,
        )
        self.assertTrue(
            _metric_warning({"source_condition_number": 81.0}, "source_condition_number", "high", 80.0)
        )

    def test_low_confidence_cases_are_recorded_without_accuracy_fields(self) -> None:
        records = [
            {
                "seed": 42,
                "true_det": 1,
                "selected_det": -1,
                "noise_level": 0.0,
                "assignment_mode": "learned_soft",
                "degeneracy_mode": "near_degenerate",
                "selected_is_correct": False,
            },
            {
                "seed": 1,
                "true_det": 1,
                "selected_det": 1,
                "noise_level": 0.0,
                "assignment_mode": "oracle_soft",
                "degeneracy_mode": "normal",
                "selected_is_correct": True,
            },
        ]
        result = _evaluate_rule(records, rule_name="test_rule", warning_fn=lambda record: record["seed"] == 42)
        self.assertEqual(result["failure_recall"], 1.0)
        self.assertEqual(len(result["low_confidence_cases"]), 1)
        self.assertNotIn("after_direction_accuracy", result["low_confidence_cases"][0])
        self.assertNotIn("after_velocity_r2", result["low_confidence_cases"][0])


if __name__ == "__main__":
    unittest.main()
