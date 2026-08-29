from __future__ import annotations

import inspect
import unittest
from dataclasses import fields
from unittest.mock import patch

import torch

import anemoi


class AttentionAPITests(unittest.TestCase):
    def test_public_api_uses_a_generic_executor(self) -> None:
        from anemoi.layers.attention import api

        signature = inspect.signature(anemoi.anemoi_attention)
        source = inspect.getsource(api)

        self.assertIn("layout", signature.parameters)
        self.assertIn("calibration", signature.parameters)
        self.assertNotIn("prefix_tokens", signature.parameters)
        self.assertNotIn("video_shape", signature.parameters)
        self.assertNotIn("anemoi.models.minimax_h3", source)

    def test_nvfp4_calibration_defaults_and_rejects_invalid_scales(self) -> None:
        self.assertEqual(anemoi.NVFP4Calibration().scales, (1.0, 1.0, 1.0))

        for value in (0.0, -1.0, float("inf"), float("nan"), True):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "finite and positive"),
            ):
                anemoi.NVFP4Calibration(q_scale=value)

    def test_calibration_keyword_requires_the_public_type(self) -> None:
        from anemoi.layers.attention.mpa import executor

        query = torch.empty((1, 64, 1, 128), dtype=torch.float16)
        with (
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            patch.object(executor, "sm120_ragged_h3_attention", return_value=query),
            self.assertRaisesRegex(TypeError, "NVFP4Calibration"),
        ):
            anemoi.anemoi_attention(
                query,
                query,
                query,
                layout=anemoi.VisualLayout(video_shape=(1, 8, 8)),
                layer=0,
                calibration=(),
            )

    def test_visual_layout_supports_prefix_and_visual_only_sequences(self) -> None:
        self.assertTrue(hasattr(anemoi, "VisualLayout"))

        prefixed = anemoi.VisualLayout(video_shape=(37, 24, 42), prefix_tokens=951)
        visual_only = anemoi.VisualLayout(video_shape=(1, 8, 8), prefix_tokens=0)

        self.assertEqual(prefixed.video_tokens, 37 * 24 * 42)
        self.assertEqual(prefixed.sequence_tokens, 951 + 37 * 24 * 42)
        self.assertEqual(visual_only.sequence_tokens, 64)

    def test_public_contract_uses_the_frozen_mean20_surface(self) -> None:
        self.assertTrue(callable(getattr(anemoi, "anemoi_attention", None)))
        self.assertEqual(
            tuple(field.name for field in fields(anemoi.SparseConfig)),
            ("query_block_size", "sparsity_ratio", "layer_sparsity_bands"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(anemoi.QuantConfig)),
            (
                "nvfp4_ratio",
                "int8_ratio",
                "fp16_ratio",
                "prefix_kv_precision",
                "prefix_query_precision",
            ),
        )

        sparse = anemoi.SparseConfig()
        per_layer = [sparse.sparsity_ratio] * 50
        for first, last, ratio in sparse.layer_sparsity_bands:
            per_layer[first:last] = [ratio] * (last - first)
        self.assertEqual(sparse.query_block_size, 64)
        self.assertEqual(sparse.sparsity_ratio, 0.80)
        self.assertEqual(sparse.layer_sparsity_bands[0], (2, 4, 0.88))
        self.assertEqual(sparse.layer_sparsity_bands[-1], (49, 50, 0.88))
        self.assertEqual(len(sparse.layer_sparsity_bands), 43)
        self.assertAlmostEqual(sum(per_layer[2:]) / 48, 0.80)

    def test_q64_and_q128_share_the_frozen_sm120_dispatch(self) -> None:
        from anemoi.layers.attention.mpa import executor

        query = torch.empty((1, 192, 2, 128), dtype=torch.float16)
        output = torch.empty_like(query)
        for query_block_size in (64, 128):
            with (
                self.subTest(query_block_size=query_block_size),
                patch("torch.cuda.get_device_capability", return_value=(12, 0)),
                patch.object(
                    executor, "sm120_ragged_h3_attention", return_value=output
                ) as operation,
                patch("torch.cuda.synchronize") as synchronize,
            ):
                result = anemoi.anemoi_attention(
                    query,
                    query,
                    query,
                    layout=anemoi.VisualLayout(video_shape=(1, 1, 128), prefix_tokens=64),
                    layer=37,
                    sparse_config=anemoi.SparseConfig(query_block_size=query_block_size),
                )

            self.assertIs(result, output)
            self.assertIs(operation.call_args.args[0], query)
            self.assertIs(operation.call_args.args[1], query)
            self.assertIs(operation.call_args.args[2], query)
            self.assertEqual(operation.call_args.kwargs["query_block_size"], query_block_size)
            self.assertEqual(operation.call_args.kwargs["sparsity_ratio"], 0.76)
            self.assertEqual(operation.call_args.kwargs["draftmap_proxy"], "mean")
            self.assertFalse(operation.call_args.kwargs["diag_jensen"])
            self.assertFalse(operation.call_args.kwargs["enable_anchors"])
            self.assertEqual(operation.call_args.kwargs["retained_mxfp8_ratio"], 0.0)
            synchronize.assert_not_called()

    def test_real_validated_sm120_mixed_cells_reach_the_public_executor(self) -> None:
        from anemoi.layers.attention.mpa import executor

        query = torch.empty((1, 128, 1, 128), dtype=torch.float16)
        newly_supported = (
            (64, (0.6, 0.4, 0.0)),
            (64, (0.85, 0.0, 0.15)),
            (64, (0.0, 0.85, 0.15)),
            (64, (0.6, 0.25, 0.15)),
            (128, (0.85, 0.0, 0.15)),
            (128, (0.6, 0.25, 0.15)),
        )
        for query_block_size, (nvfp4, int8, fp16) in newly_supported:
            output = torch.empty_like(query)
            prefix_precision = "int8" if int8 else "fp16"
            with (
                self.subTest(query_block_size=query_block_size, ratios=(nvfp4, int8, fp16)),
                patch("torch.cuda.get_device_capability", return_value=(12, 0)),
                patch.object(
                    executor, "sm120_ragged_h3_attention", return_value=output
                ) as operation,
            ):
                result = anemoi.anemoi_attention(
                    query,
                    query,
                    query,
                    layout=anemoi.VisualLayout(video_shape=(1, 1, 128)),
                    layer=25,
                    sparse_config=anemoi.SparseConfig(query_block_size=query_block_size),
                    quant_config=anemoi.QuantConfig(
                        nvfp4_ratio=nvfp4,
                        int8_ratio=int8,
                        fp16_ratio=fp16,
                        prefix_kv_precision=prefix_precision,
                        prefix_query_precision=prefix_precision,
                    ),
                )
                self.assertIs(result, output)
                operation.assert_called_once()

    def test_nvfp4_dispatch_uses_unity_default_and_independent_explicit_scales(self) -> None:
        from anemoi.layers.attention.mpa import executor

        query = torch.empty((1, 192, 1, 128), dtype=torch.float16)
        output = torch.empty_like(query)
        cases = (
            (None, (1.0, 1.0, 1.0)),
            (anemoi.NVFP4Calibration(2.0, 3.0, 4.0), (2.0, 3.0, 4.0)),
        )
        for calibration, expected in cases:
            with (
                self.subTest(calibration=calibration),
                patch("torch.cuda.get_device_capability", return_value=(12, 0)),
                patch.object(
                    executor, "sm120_ragged_h3_attention", return_value=output
                ) as operation,
            ):
                result = anemoi.anemoi_attention(
                    query,
                    query,
                    query,
                    layout=anemoi.VisualLayout(video_shape=(1, 1, 128), prefix_tokens=64),
                    layer=2,
                    sparse_config=anemoi.SparseConfig(query_block_size=64),
                    quant_config=anemoi.QuantConfig(
                        nvfp4_ratio=1.0,
                        int8_ratio=0.0,
                        prefix_kv_precision="nvfp4",
                        prefix_query_precision="fp16",
                    ),
                    calibration=calibration,
                )

            self.assertIs(result, output)
            self.assertEqual(operation.call_args.kwargs["nvfp4_scales"], expected)

    def test_nvfp4_scale_tensor_cache_reuses_allocations(self) -> None:
        from anemoi.layers.attention.mpa import executor

        executor._nvfp4_global_scales.cache_clear()
        tensor = torch.tensor
        with patch.object(executor.torch, "tensor", side_effect=tensor) as allocate:
            first = executor._nvfp4_global_scales(torch.device("cpu"), (2.0, 3.0, 4.0))
            second = executor._nvfp4_global_scales(torch.device("cpu"), (2.0, 3.0, 4.0))

        self.assertIs(first, second)
        self.assertEqual(allocate.call_count, 3)

    def test_unsupported_flash_options_fail_before_backend_dispatch(self) -> None:
        from anemoi.layers.attention.mpa import executor

        query = torch.empty((1, 192, 1, 128), dtype=torch.float16)
        cases = (
            {"attn_mask": torch.ones(1, dtype=torch.bool)},
            {"dropout_p": 0.1},
            {"is_causal": True},
            {"scale": 1.0},
        )
        for kwargs in cases:
            with (
                self.subTest(kwargs=kwargs),
                patch("torch.cuda.get_device_capability", return_value=(12, 0)),
                patch.object(
                    executor, "sm120_ragged_h3_attention", return_value=query
                ) as operation,
                self.assertRaises(ValueError),
            ):
                anemoi.anemoi_attention(
                    query,
                    query,
                    query,
                    layout=anemoi.VisualLayout(video_shape=(1, 1, 128), prefix_tokens=64),
                    layer=2,
                    **kwargs,
                )
            operation.assert_not_called()

    def test_public_configs_reject_invalid_cells(self) -> None:
        from anemoi.layers.attention.mpa import executor

        with self.assertRaisesRegex(ValueError, "query_block_size"):
            anemoi.SparseConfig(query_block_size=96)
        with self.assertRaisesRegex(ValueError, "sorted, disjoint"):
            anemoi.SparseConfig(layer_sparsity_bands=((2, 5, 0.8), (4, 6, 0.8)))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            anemoi.QuantConfig(int8_ratio=0.5)

        query = torch.empty((1, 192, 1, 128), dtype=torch.float16)
        with (
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            patch.object(executor, "sm120_ragged_h3_attention") as operation,
            self.assertRaisesRegex(RuntimeError, "prefix-query INT8"),
        ):
            anemoi.anemoi_attention(
                query,
                query,
                query,
                layout=anemoi.VisualLayout(video_shape=(1, 1, 128), prefix_tokens=64),
                layer=2,
                quant_config=anemoi.QuantConfig(
                    int8_ratio=0.0,
                    fp16_ratio=1.0,
                    prefix_kv_precision="fp16",
                ),
            )
        operation.assert_not_called()

    def test_sm89_reuses_the_same_public_entry(self) -> None:
        from anemoi.layers.attention.mpa import executor

        query = torch.empty((1, 192, 1, 128), dtype=torch.float16)
        output = torch.empty_like(query)
        with (
            patch("torch.cuda.get_device_capability", return_value=(8, 9)),
            patch.object(executor, "sm89_ragged_h3_attention", return_value=output) as operation,
        ):
            result = anemoi.anemoi_attention(
                query,
                query,
                query,
                layout=anemoi.VisualLayout(video_shape=(1, 1, 128), prefix_tokens=64),
                layer=2,
            )

        self.assertIs(result, output)
        self.assertIs(operation.call_args.args[0], query)
        self.assertEqual(operation.call_args.kwargs["retained_int8_ratio"], 1.0)
        self.assertEqual(operation.call_args.kwargs["retained_fp16_ratio"], 0.0)
        self.assertFalse(operation.call_args.kwargs["diag_jensen"])
        self.assertFalse(operation.call_args.kwargs["enable_anchors"])

        with (
            patch("torch.cuda.get_device_capability", return_value=(8, 9)),
            patch.object(executor, "sm89_ragged_h3_attention") as operation,
            self.assertRaisesRegex(RuntimeError, "SM89 supports"),
        ):
            anemoi.anemoi_attention(
                query,
                query,
                query,
                layout=anemoi.VisualLayout(video_shape=(1, 1, 128), prefix_tokens=64),
                layer=2,
                sparse_config=anemoi.SparseConfig(query_block_size=128),
            )
        operation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
