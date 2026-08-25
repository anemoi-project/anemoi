from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from anemoi.models.minimax_h3.mpa_attention import H3MPAConfig
from anemoi.models.minimax_h3.runner import (
    CANDIDATES,
    DEFAULT_MPA_CONFIG,
    MPA_MAINLINE_CANDIDATE,
    _read_mpa_config,
    _resolved_candidate,
)


class MiniMaxH3MPAConfigTests(unittest.TestCase):
    def test_launcher_defaults_sm120_to_q64_int8(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1] / "scripts/run_minimax_h3.sh"
        ).read_text()

        self.assertIn("mpa_config_set=0", launcher)
        self.assertIn("mpa-sm120-q64-int8.yaml", launcher)

    def test_released_sm89_policy_keeps_ragged_anchors(self) -> None:
        config = _read_mpa_config(DEFAULT_MPA_CONFIG)
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            mpa_config=config,
        )

        self.assertIs(config["enable_anchors"], True)
        self.assertTrue(policy["enable_anchors"])
        self.assertTrue(H3MPAConfig().enable_anchors)

    def test_prefix_precision_defaults_to_auto(self) -> None:
        config = H3MPAConfig()
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            mpa_config=_read_mpa_config(DEFAULT_MPA_CONFIG),
        )

        self.assertEqual(config.prefix_kv_precision, "auto")
        self.assertEqual(config.prefix_query_precision, "auto")
        self.assertEqual(policy["prefix_kv_precision"], "auto")
        self.assertEqual(policy["prefix_query_precision"], "auto")

    def test_explicit_prefix_precision_is_preserved_in_policy(self) -> None:
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            mpa_config={
                "query_block_size": 128,
                "nvfp4_ratio": 0.6,
                "int8_ratio": 0.0,
                "mxfp8_ratio": 0.4,
                "fp16_ratio": 0.0,
                "prefix_kv_precision": "mxfp8",
                "prefix_query_precision": "int8",
            },
        )

        self.assertEqual(policy["prefix_kv_precision"], "mxfp8")
        self.assertEqual(policy["prefix_query_precision"], "int8")

    def test_unknown_prefix_precision_is_rejected(self) -> None:
        for field in ("prefix_kv_precision", "prefix_query_precision"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                _resolved_candidate(
                    MPA_MAINLINE_CANDIDATE,
                    mpa_config={field: "fp8"},
                )

    def test_public_candidates_do_not_expose_unbundled_comparators(self) -> None:
        self.assertEqual(
            CANDIDATES,
            ("dense", "official-sol", MPA_MAINLINE_CANDIDATE),
        )

    def test_q128_nv75_int25_policy_is_accepted(self) -> None:
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            mpa_config={
                "query_block_size": 128,
                "prefix_kv_precision": "int8",
                "nvfp4_ratio": 0.75,
                "int8_ratio": 0.25,
                "fp16_ratio": 0.0,
            },
        )

        self.assertEqual(
            policy["precision"],
            {"nvfp4": 0.75, "int8": 0.25, "mxfp8": 0.0, "fp16": 0.0},
        )
        self.assertEqual(policy["query_block_size"], 128)
        self.assertEqual(policy["prefix_kv_precision"], "int8")

    def test_sm120_q128_public_presets_resolve(self) -> None:
        expected = {
            "mpa-sm120-q128-int8.yaml": {
                "nvfp4": 0.0,
                "int8": 1.0,
                "mxfp8": 0.0,
                "fp16": 0.0,
            },
            "mpa-sm120-q128-nv60-int40.yaml": {
                "nvfp4": 0.60,
                "int8": 0.40,
                "mxfp8": 0.0,
                "fp16": 0.0,
            },
            "mpa-sm120-q128-nv75-int25.yaml": {
                "nvfp4": 0.75,
                "int8": 0.25,
                "mxfp8": 0.0,
                "fp16": 0.0,
            },
        }
        root = Path(__file__).resolve().parents[1] / "examples/minimax-h3"

        for name, precision in expected.items():
            policy = _resolved_candidate(
                MPA_MAINLINE_CANDIDATE,
                mpa_config=_read_mpa_config(root / name),
            )
            self.assertEqual(policy["precision"], precision)
            self.assertEqual(policy["query_block_size"], 128)
            self.assertEqual(policy["prefix_kv_precision"], "int8")

    def test_sm120_q64_production_preset_enables_anchors(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples/minimax-h3"
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            mpa_config=_read_mpa_config(root / "mpa-sm120-q64-int8.yaml"),
        )

        self.assertEqual(policy["query_block_size"], 64)
        self.assertEqual(policy["prefix_kv_precision"], "int8")
        self.assertEqual(policy["prefix_query_precision"], "int8")
        self.assertEqual(
            policy["precision"],
            {"nvfp4": 0.0, "int8": 1.0, "mxfp8": 0.0, "fp16": 0.0},
        )
        self.assertTrue(policy["enable_anchors"])

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
