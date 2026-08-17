from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evg.models.minimax_h3.runner import (
    DEFAULT_MPA_CONFIG,
    MPA_MAINLINE_CANDIDATE,
    _read_mpa_config,
    _resolved_candidate,
)


class MiniMaxH3MPAConfigTests(unittest.TestCase):
    def test_released_yaml_resolves_to_the_released_policy(self) -> None:
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            mpa_config=_read_mpa_config(DEFAULT_MPA_CONFIG),
        )

        self.assertEqual(policy["precision"], {"fp8": 0.8, "fp16": 0.2})
        self.assertEqual(policy["video_sparsity_ratio"], 0.88)
        self.assertEqual(
            policy["layer_sparsity_bands"],
            ((18, 34, 0.82), (34, 50, 0.58)),
        )
        self.assertAlmostEqual(policy["average_sparse_layer_sparsity_ratio"], 0.76)
        self.assertEqual(policy["dense_first_steps"], 10)
        self.assertEqual(policy["dense_first_layers"], 2)
        self.assertFalse(policy["diag_jensen"])

    def test_yaml_changes_policy_and_cli_precision_has_priority(self) -> None:
        config = {
            "sparsity_ratio": 0.75,
            "layer_sparsity_bands": [],
            "fp8_ratio": 0.6,
            "fp16_ratio": 0.4,
            "dense_first_steps": 4,
            "dense_first_layers": 1,
            "diag_jensen": True,
        }
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            fp8_ratio=0.7,
            fp16_ratio=0.3,
            mpa_config=config,
        )

        self.assertEqual(policy["precision"], {"fp8": 0.7, "fp16": 0.3})
        self.assertEqual(policy["video_sparsity_ratio"], 0.75)
        self.assertEqual(policy["layer_sparsity_bands"], ())
        self.assertEqual(policy["dense_first_steps"], 4)
        self.assertEqual(policy["dense_first_layers"], 1)
        self.assertTrue(policy["diag_jensen"])

    def test_unknown_yaml_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mpa.yaml"
            path.write_text("sparsity: 0.9\n")
            with self.assertRaisesRegex(ValueError, "unknown MPA config keys"):
                _read_mpa_config(path)

    def test_diag_jensen_requires_a_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "diag_jensen must be a boolean"):
            _resolved_candidate(
                MPA_MAINLINE_CANDIDATE,
                mpa_config={"diag_jensen": "false"},
            )

    def test_overlapping_layer_bands_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted, disjoint"):
            _resolved_candidate(
                MPA_MAINLINE_CANDIDATE,
                mpa_config={
                    "layer_sparsity_bands": [[10, 20, 0.8], [19, 30, 0.7]]
                },
            )


if __name__ == "__main__":
    unittest.main()
