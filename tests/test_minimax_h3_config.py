from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from anemoi import SparseConfig
from anemoi.models.minimax_h3.mpa_attention import H3MPAConfig
from anemoi.models.minimax_h3.runner import (
    CANDIDATES,
    DEFAULT_MPA_CONFIG,
    MPA_MAINLINE_CANDIDATE,
    _attention_precision_note,
    _read_experimental_mpa_config,
    _read_mpa_config,
    _resolved_candidate,
)


class MiniMaxH3MPAConfigTests(unittest.TestCase):
    def test_launcher_uses_one_default_config_for_sm89_and_sm120(self) -> None:
        launcher = (Path(__file__).resolve().parents[1] / "scripts/run_minimax_h3.sh").read_text()

        self.assertNotIn("mpa_config_set", launcher)
        self.assertNotIn("mpa-sm120-q64-int8.yaml", launcher)

    def test_released_policy_freezes_mean_without_jensen_or_anchors(self) -> None:
        config = _read_mpa_config(DEFAULT_MPA_CONFIG)
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            mpa_config=config,
        )

        self.assertNotIn("diag_jensen", config)
        self.assertNotIn("enable_anchors", config)
        self.assertNotIn("mxfp8_ratio", config)
        self.assertFalse(policy["diag_jensen"])
        self.assertFalse(policy["enable_anchors"])
        self.assertFalse(H3MPAConfig().diag_jensen)
        self.assertFalse(H3MPAConfig().enable_anchors)

    def test_released_route_note_describes_active_routing(self) -> None:
        policy = _resolved_candidate(MPA_MAINLINE_CANDIDATE)

        self.assertEqual(
            policy["route_note"],
            "Mean pooled-QK routing with exact per-head global top-k",
        )

    def test_custom_sparsity_route_note_remains_schedule_neutral(self) -> None:
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            mpa_config={"sparsity_ratio": 0.75, "layer_sparsity_bands": []},
        )

        self.assertEqual(
            policy["route_note"],
            "Mean pooled-QK routing with exact per-head global top-k",
        )

    def test_released_skip_policy_describes_active_behavior(self) -> None:
        policy = _resolved_candidate(MPA_MAINLINE_CANDIDATE)

        self.assertEqual(policy["skip_compensation"], "unselected blocks are dropped")

    def test_default_precision_is_portable_q64_pure_int8(self) -> None:
        config = H3MPAConfig()
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            mpa_config=_read_mpa_config(DEFAULT_MPA_CONFIG),
        )

        self.assertEqual(config.prefix_kv_precision, "int8")
        self.assertEqual(config.prefix_query_precision, "int8")
        self.assertEqual(config.precision(0), (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(config.sm89_precision(0), (1.0, 0.0))
        self.assertEqual(config.draftmap(0), "mean")
        self.assertEqual(policy["prefix_kv_precision"], "int8")
        self.assertEqual(policy["prefix_query_precision"], "int8")
        self.assertEqual(
            policy["precision"],
            {"nvfp4": 0.0, "int8": 1.0, "mxfp8": 0.0, "fp16": 0.0},
        )
        self.assertNotIn("draftmap_proxy", policy)
        self.assertNotIn("layer_draftmap_bands", policy)

        note = _attention_precision_note(policy)
        self.assertIn("prefix queries use INT8", note)
        self.assertIn("prefix K/V uses INT8", note)
        self.assertNotIn("prefix-query overwrites use the original-dtype", note)
        self.assertNotIn("prefix K/V is FP16", note)

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
            self.assertEqual(
                policy["layer_sparsity_bands"],
                SparseConfig().layer_sparsity_bands,
            )

    def test_sm120_q64_production_preset_freezes_mean_without_anchors(self) -> None:
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
        self.assertFalse(policy["diag_jensen"])
        self.assertFalse(policy["enable_anchors"])

    def test_q64_default_and_architecture_aliases_are_logically_identical(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1] / "examples/minimax-h3"
        names = (
            "mpa-ragged2d-mixed.yaml",
            "mpa-sm120-q64-int8.yaml",
        )
        policies = [
            _resolved_candidate(
                MPA_MAINLINE_CANDIDATE,
                mpa_config=_read_mpa_config(root / name),
            )
            for name in names
        ]

        for policy in policies[1:]:
            self.assertEqual(policy, policies[0])
        self.assertEqual(policies[0]["query_block_size"], 64)
        self.assertEqual(policies[0]["prefix_kv_precision"], "int8")
        self.assertEqual(policies[0]["prefix_query_precision"], "int8")
        self.assertFalse(policies[0]["enable_anchors"])
        self.assertNotIn("draftmap_proxy", policies[0])
        self.assertNotIn("layer_draftmap_bands", policies[0])

    def test_released_yaml_resolves_to_the_released_policy(self) -> None:
        config = _read_mpa_config(DEFAULT_MPA_CONFIG)
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            mpa_config=config,
        )

        self.assertNotIn("sparsity_ratio", config)
        self.assertNotIn("layer_sparsity_bands", config)
        self.assertEqual(
            policy["precision"],
            {"nvfp4": 0.0, "int8": 1.0, "mxfp8": 0.0, "fp16": 0.0},
        )
        self.assertEqual(policy["video_sparsity_ratio"], 0.80)
        self.assertEqual(
            policy["layer_sparsity_bands"],
            SparseConfig().layer_sparsity_bands,
        )
        self.assertAlmostEqual(policy["average_sparse_layer_sparsity_ratio"], 0.80)
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

    def test_stable_yaml_rejects_experimental_fields(self) -> None:
        experimental = (
            "fp8_ratio",
            "mxfp8_ratio",
            "draftmap_proxy",
            "layer_precision_bands",
            "layer_draftmap_bands",
            "diag_jensen",
            "enable_anchors",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mpa.yaml"
            for field in experimental:
                with self.subTest(field=field):
                    path.write_text(f"{field}: false\n")
                    with self.assertRaisesRegex(ValueError, "stable MPA config"):
                        _read_mpa_config(path)

    def test_experimental_yaml_parser_keeps_internal_replay_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mpa.yaml"
            path.write_text("draftmap_proxy: k_tail_r1\nenable_anchors: true\n")

            self.assertEqual(
                _read_experimental_mpa_config(path),
                {"draftmap_proxy": "k_tail_r1", "enable_anchors": True},
            )

    def test_stable_yaml_rejects_experimental_prefix_precision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mpa.yaml"
            path.write_text("prefix_kv_precision: mxfp8\n")
            with self.assertRaisesRegex(ValueError, "prefix_kv_precision"):
                _read_mpa_config(path)

    def test_diag_jensen_requires_a_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "diag_jensen must be a boolean"):
            _resolved_candidate(
                MPA_MAINLINE_CANDIDATE,
                mpa_config={"diag_jensen": "false"},
            )

    def test_k_tail_layer_bands_resolve_non_contiguous_ranks(self) -> None:
        bands = ((38, 39, "k_tail_r1"), (42, 43, "k_tail_r2"))
        config = H3MPAConfig(
            query_block_size=64,
            fp8_ratio=0.0,
            int8_ratio=1.0,
            fp16_ratio=0.0,
            draftmap_proxy="mean",
            layer_draftmap_bands=bands,
        )
        policy = _resolved_candidate(
            MPA_MAINLINE_CANDIDATE,
            mpa_config={
                "query_block_size": 64,
                "nvfp4_ratio": 0.0,
                "int8_ratio": 1.0,
                "fp16_ratio": 0.0,
                "draftmap_proxy": "mean",
                "layer_draftmap_bands": [list(band) for band in bands],
            },
        )

        self.assertEqual(
            tuple(config.draftmap(layer) for layer in (37, 38, 39, 42, 43)),
            ("mean", "k_tail_r1", "mean", "k_tail_r2", "mean"),
        )
        self.assertEqual(policy["draftmap_proxy"], "mean")
        self.assertEqual(policy["layer_draftmap_bands"], bands)

    def test_k_tail_rejects_unsupported_configurations(self) -> None:
        sm120 = {
            "query_block_size": 64,
            "fp8_ratio": 0.0,
            "int8_ratio": 1.0,
            "fp16_ratio": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "draftmap_proxy"):
            H3MPAConfig(**sm120, draftmap_proxy="other")
        with self.assertRaisesRegex(ValueError, "sorted, disjoint"):
            H3MPAConfig(
                **sm120,
                layer_draftmap_bands=(
                    (10, 20, "k_tail_r2"),
                    (19, 21, "mean"),
                ),
            )
        self.assertEqual(H3MPAConfig(draftmap_proxy="k_tail_r1").draftmap(0), "k_tail_r1")
        with self.assertRaisesRegex(ValueError, "portable Q64"):
            H3MPAConfig(
                **{**sm120, "query_block_size": 128},
                draftmap_proxy="k_tail_r1",
            )
        with self.assertRaisesRegex(ValueError, "portable Q64"):
            H3MPAConfig(
                fp8_ratio=0.8,
                fp16_ratio=0.2,
                draftmap_proxy="k_tail_r1",
            )
        with self.assertRaisesRegex(ValueError, "diag_jensen"):
            H3MPAConfig(**sm120, draftmap_proxy="k_tail_r1", diag_jensen=True)

    def test_overlapping_layer_bands_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted, disjoint"):
            _resolved_candidate(
                MPA_MAINLINE_CANDIDATE,
                mpa_config={"layer_sparsity_bands": [[10, 20, 0.8], [19, 30, 0.7]]},
            )


if __name__ == "__main__":
    unittest.main()
