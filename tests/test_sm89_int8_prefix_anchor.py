from __future__ import annotations

import math
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]


class SM89Int8PrefixBackendTests(unittest.TestCase):
    def test_prefix_producer_dispatches_only_to_the_native_operator(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm89_k64

        operation = Mock(return_value=("q8", "scale"))
        ops = SimpleNamespace(prepare_prefix_q_int8=operation)
        query = torch.empty((1, 2, 95, 64), dtype=torch.float16)

        with patch.object(sm89_k64, "_load_k64_ops", return_value=ops):
            result = sm89_k64.prepare_prefix_q_int8(query, 95)

        self.assertEqual(result, ("q8", "scale"))
        operation.assert_called_once_with(query, 95)

    def test_prefix_wrapper_reuses_shared_k_and_v_operands(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm89_k64

        operation = Mock(return_value="prefix-output")
        ops = SimpleNamespace(prefix_int8_attention=operation)
        prefix_q8 = torch.empty((1, 14, 64, 128), dtype=torch.int8)
        prefix_q_scale = torch.empty((1, 14, 1), dtype=torch.float32)
        shared = tuple(torch.tensor(index) for index in range(6))
        valid = torch.full((1, 2), 64, dtype=torch.int32)

        with patch.object(sm89_k64, "_load_k64_ops", return_value=ops):
            result = sm89_k64.sm89_q64_prefix_int8_attention(
                prefix_q8, prefix_q_scale, shared, valid, 61
            )

        self.assertEqual(result, "prefix-output")
        self.assertEqual(
            operation.call_args.args,
            (
                prefix_q8,
                shared[1],
                shared[2],
                prefix_q_scale,
                shared[4],
                shared[5],
                valid,
                61,
                1.0 / math.sqrt(128),
            ),
        )

    def test_mixed_wrapper_accepts_prepared_operands_without_requantizing(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm89_k64

        operation = Mock(return_value=("output", "lse"))
        ops = SimpleNamespace(mixed_attention=operation)
        query = torch.empty((1, 2, 64, 128), dtype=torch.float16)
        key = value = torch.empty((1, 2, 128, 128), dtype=torch.float16)
        route = torch.zeros((1, 2, 1, 2), dtype=torch.int32)
        low = high = torch.ones((1, 2, 1), dtype=torch.int32)
        valid = torch.full((1, 2), 64, dtype=torch.int32)
        prepared = tuple(torch.tensor(index) for index in range(6))

        with patch.object(sm89_k64, "_load_k64_ops", return_value=ops), patch.object(
            sm89_k64,
            "prepare_k64_fp8_operands",
            side_effect=AssertionError("shared preparation must be reused"),
        ):
            result = sm89_k64.native_k64_mixed_attention(
                query,
                key,
                value,
                route,
                low,
                route,
                high,
                valid,
                prepared_operands=prepared,
        )

        self.assertEqual(result, ("output", "lse"))
        arguments = operation.call_args.args
        for actual, expected in zip(arguments[:3], prepared[:3]):
            self.assertIs(actual, expected)
        for actual, expected in zip(arguments[3:6], (query, key, value)):
            self.assertIs(actual, expected)
        for actual, expected in zip(arguments[10:13], prepared[3:]):
            self.assertIs(actual, expected)

    def test_pure_int8_wrapper_selects_the_no_fp16_specialization(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm89_k64

        int8_operation = Mock(return_value=("output", "lse"))
        mixed_operation = Mock(side_effect=AssertionError("mixed phase is inactive"))
        ops = SimpleNamespace(
            int8_attention=int8_operation,
            mixed_attention=mixed_operation,
        )
        query = torch.empty((1, 2, 64, 128), dtype=torch.float16)
        key = value = torch.empty((1, 2, 128, 128), dtype=torch.float16)
        route = torch.zeros((1, 2, 1, 2), dtype=torch.int32)
        counts = torch.ones((1, 2, 1), dtype=torch.int32)
        valid = torch.full((1, 2), 64, dtype=torch.int32)
        prepared = tuple(torch.tensor(index) for index in range(6))

        with patch.object(sm89_k64, "_load_k64_ops", return_value=ops):
            result = sm89_k64.native_k64_mixed_attention(
                query,
                key,
                value,
                route,
                counts,
                route,
                torch.zeros_like(counts),
                valid,
                prepared_operands=prepared,
                active_fp16=False,
            )

        self.assertEqual(result, ("output", "lse"))
        mixed_operation.assert_not_called()
        self.assertIs(int8_operation.call_args.args[0], prepared[0])
        self.assertIs(int8_operation.call_args.args[1], prepared[1])
        self.assertIs(int8_operation.call_args.args[2], prepared[2])
        self.assertIs(int8_operation.call_args.args[3], route)
        self.assertIs(int8_operation.call_args.args[4], counts)

    def test_pure_fp16_wrapper_skips_int8_preparation_and_mixed_kernel(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm89_k64

        fp16_operation = Mock(return_value=("output", "lse"))
        mixed_operation = Mock(side_effect=AssertionError("mixed phase is inactive"))
        ops = SimpleNamespace(
            fp16_attention=fp16_operation,
            mixed_attention=mixed_operation,
        )
        query = torch.empty((1, 2, 64, 128), dtype=torch.float16)
        key = value = torch.empty((1, 2, 128, 128), dtype=torch.float16)
        route = torch.zeros((1, 2, 1, 2), dtype=torch.int32)
        counts = torch.ones((1, 2, 1), dtype=torch.int32)
        valid = torch.full((1, 2), 64, dtype=torch.int32)

        with patch.object(sm89_k64, "_load_k64_ops", return_value=ops), patch.object(
            sm89_k64,
            "prepare_k64_fp8_operands",
            side_effect=AssertionError("pure FP16 must not prepare INT8 operands"),
        ):
            result = sm89_k64.native_k64_mixed_attention(
                query,
                key,
                value,
                route,
                torch.zeros_like(counts),
                route,
                counts,
                valid,
                active_int8=False,
            )

        self.assertEqual(result, ("output", "lse"))
        mixed_operation.assert_not_called()
        self.assertEqual(
            fp16_operation.call_args.args,
            (query, key, value, route, counts, valid, 1.0 / math.sqrt(128)),
        )

    def test_route_wrapper_forwards_fixed_budget_anchors(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm89_k64

        operation = Mock(return_value=("ids", "low", "middle", "high"))
        ops = SimpleNamespace(route_precision=operation)
        probability = torch.empty((1, 14, 32, 32), dtype=torch.float16)
        anchors = torch.eye(32, dtype=torch.bool)
        anchor_ids = anchors.flatten().nonzero().flatten().to(torch.int32)

        with patch.object(sm89_k64, "_load_k64_ops", return_value=ops):
            result = sm89_k64.sm89_h3_route_precision(
                probability,
                n16=3,
                n8=5,
                anchors=anchors,
                anchor_ids=anchor_ids,
                anchor_count=32,
            )

        self.assertEqual(result, ("ids", "low", "middle", "high"))
        self.assertIs(operation.call_args.args[0], probability)
        self.assertEqual(operation.call_args.args[1:4], (3, 5, 0))
        self.assertIs(operation.call_args.args[4], anchors)
        self.assertIs(operation.call_args.args[5], anchor_ids)
        self.assertEqual(operation.call_args.args[6], 32)

    def test_materializer_selects_explicit_int8_or_implicit_fp16_prefix(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm89_k64

        logical = torch.zeros((1, 2, 3, 3), dtype=torch.int32)
        counts = torch.ones((1, 2, 3), dtype=torch.int32)
        native_low = torch.zeros_like(counts)
        native_middle = torch.ones_like(counts)
        native_high = torch.ones_like(counts)
        operation = Mock(
            return_value=(logical, native_low, native_middle, native_high)
        )
        ops = SimpleNamespace(materialize_route=operation)

        with patch.object(sm89_k64, "_load_k64_ops", return_value=ops):
            int8 = sm89_k64.sm89_h3_materialize_route(
                logical,
                native_low,
                counts,
                counts,
                prefix_blocks=2,
                prefix_int8=True,
            )
            fp16 = sm89_k64.sm89_h3_materialize_route(
                logical,
                native_low,
                counts,
                counts,
                prefix_blocks=2,
                prefix_int8=False,
            )

        self.assertEqual(int8, (logical, native_middle, native_high))
        self.assertEqual(fp16, int8)
        self.assertEqual(operation.call_args_list[0].args[-5:], (64, 2, 1, True, True))
        self.assertEqual(operation.call_args_list[1].args[-5:], (64, 2, 2, False, True))

    def test_output_assembly_preserves_requested_dtype(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm89_k64

        operation = Mock(return_value="output")
        ops = SimpleNamespace(assemble_h3_output=operation)
        prefix = torch.empty((1, 2, 64, 128), dtype=torch.bfloat16)
        video = torch.empty((1, 2, 64, 128), dtype=torch.float16)
        inverse = torch.arange(64, dtype=torch.int64)
        counts = (
            torch.zeros((1, 2, 1), dtype=torch.int32),
            torch.zeros((1, 2, 1), dtype=torch.int32),
            None,
        )

        with patch.object(sm89_k64, "_load_k64_ops", return_value=ops):
            result = sm89_k64.assemble_h3_k64_output(
                prefix,
                video,
                inverse,
                output_dtype=torch.bfloat16,
                route_counts=counts,
                query_block_size=64,
            )

        self.assertEqual(result, "output")
        self.assertEqual(
            operation.call_args.args,
            (prefix, video, inverse, torch.bfloat16, *counts, 64),
        )

        operation.reset_mock()
        with patch.object(sm89_k64, "_load_k64_ops", return_value=ops):
            sm89_k64.assemble_h3_k64_output(prefix, video, inverse)
        self.assertEqual(
            operation.call_args.args,
            (prefix, video, inverse, None, None, None, None, 0),
        )


class SM89Int8PrefixBuildTests(unittest.TestCase):
    def test_sm89_empty_output_cleanup_is_owned_by_final_assembly(self) -> None:
        executor = (
            ROOT / "anemoi/layers/attention/mpa/executor.py"
        ).read_text()
        mainloop = (
            ROOT / "csrc/attention/cuda/sm89/mixed_attention.cuh"
        ).read_text()

        self.assertIn(
            "route_counts=(fp8_counts, fp16_counts, None)", executor
        )
        empty_start = mainloop.index("if (__builtin_expect(")
        empty_end = mainloop.index("const float base2_softmax_scale", empty_start)
        empty_path = mainloop[empty_start:empty_end]
        self.assertIn("return;", empty_path)
        self.assertIn("fp16_prefix_stages == 0xffffffffu", empty_path)
        self.assertIn("output_words", empty_path)
        self.assertIn("lse[", empty_path)

        host_source = (
            ROOT / "csrc/attention/cuda/sm89/k64_attention_host.cu"
        ).read_text()
        self.assertIn(
            "fp16_prefix_blocks >= 0 && fp16_prefix_blocks <= key_blocks",
            host_source,
        )

    def test_sm89_sparse_attention_disables_lse_writeback_at_runtime(self) -> None:
        kernel = (
            ROOT / "csrc/attention/cuda/sm89/mixed_attention.cuh"
        ).read_text()
        host = (
            ROOT / "csrc/attention/cuda/sm89/k64_attention_host.cu"
        ).read_text()

        self.assertGreaterEqual(kernel.count("lse != nullptr"), 2)
        self.assertIn("lse[(batch_id * num_qo_heads", kernel)
        self.assertNotIn("lse.data_ptr", host)
        self.assertGreaterEqual(
            len(re.findall(
                r"valid_k_counts\.data_ptr<int32_t>\(\),\s*nullptr,", host
            )),
            6,
        )
        self.assertEqual(
            len(re.findall(r"auto lse = torch::empty\(\s*\{0\},", host)),
            3,
        )
        self.assertNotIn("add_smoothed_lse_offset_kernel", host)

    def test_sm89_build_registers_route_and_dense_prefix_sources(self) -> None:
        setup = (ROOT / "setup.py").read_text()
        api = (ROOT / "csrc/attention/cuda/sm89/api.h").read_text()
        bindings = (ROOT / "csrc/attention/cuda/sm89/bindings.cpp").read_text()

        self.assertIn('_source(sm89 / "h3_route_precision.cu")', setup)
        self.assertIn("inst_k64_d128_int8_dense.cu", setup)
        for source in (api, bindings):
            self.assertIn("sm89_h3_route_precision", source)
            self.assertIn("sm89_h3_materialize_route", source)
            self.assertIn("prepare_sm89_prefix_q_int8", source)
            self.assertIn("sm89_q64_prefix_int8_attention", source)

        backend = (
            ROOT / "anemoi/layers/attention/mpa/backends/sm89_k64.py"
        ).read_text()
        producer_start = backend.index("def prepare_prefix_q_int8(")
        producer_end = backend.index("\ndef sm89_q64_prefix_int8_attention(", producer_start)
        self.assertNotIn("_sm89_qk_quant", backend[producer_start:producer_end])

    def test_sm89_route_reuses_the_verified_native_algorithm(self) -> None:
        route = (ROOT / "csrc/attention/cuda/sm89/h3_route_precision.cu").read_text()
        donor = (ROOT / "csrc/attention/cuda/sm120/h3_route_precision.cu").read_text()

        self.assertIn('#include "../sm120/h3_route_precision.cu"', route)
        self.assertIn("MPA_IMPLICIT_HIGH_PREFIX 1", route)
        self.assertIn("apply_anchor_budget_kernel", donor)
        self.assertIn("pack_active_rows_kernel", donor)
        self.assertIn("cub::DeviceRadixSort::SortPairs", donor)

    def test_dense_prefix_instantiation_uses_production_phase_without_a_route(self) -> None:
        instantiation = (
            ROOT
            / "csrc/attention/cuda/sm89/instantiations/inst_k64_d128_int8_dense.cu"
        ).read_text()
        mainloop = (ROOT / "csrc/attention/cuda/sm89/mixed_attention.cuh").read_text()

        self.assertIn("#define MPA_DENSE_SEQUENTIAL 1", instantiation)
        self.assertIn("#define MPA_STORE_LSE 0", instantiation)
        self.assertIn("num_physical_stages", mainloop)
        dense_start = mainloop.index("#if MPA_DENSE_SEQUENTIAL")
        dense_end = mainloop.index("#else", dense_start)
        self.assertNotIn("block_ids", mainloop[dense_start:dense_end])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SM89NativeRouteParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if torch.cuda.get_device_capability() != (8, 9):
            raise unittest.SkipTest("SM89 is required")
        from anemoi.layers.attention.mpa.backends import sm89_k64

        try:
            sm89_k64._load_k64_ops()
        except (ImportError, OSError, RuntimeError) as exc:
            raise unittest.SkipTest(f"SM89 extension is unavailable: {exc}") from exc

    def test_native_prefix_quantization_matches_the_frozen_contract(self) -> None:
        from anemoi.layers.attention.mpa.backends.sm89_k64 import (
            prepare_prefix_q_int8,
        )

        prefix_tokens = 95
        for dtype in (torch.float16, torch.bfloat16):
            for head_dim in (64, 128):
                with self.subTest(dtype=dtype, head_dim=head_dim):
                    torch.manual_seed(1000 + head_dim)
                    source = torch.randn(
                        (2, prefix_tokens + 7, 3, head_dim),
                        device="cuda",
                        dtype=dtype,
                    )
                    query = source.permute(0, 2, 1, 3)
                    actual, scales = prepare_prefix_q_int8(query, prefix_tokens)
                    expected_values = torch.zeros(
                        (2, 3, 128, head_dim),
                        device="cuda",
                        dtype=torch.float32,
                    )
                    expected_values[:, :, :prefix_tokens] = query[
                        :, :, :prefix_tokens
                    ].float()
                    blocks = expected_values.view(2, 3, 2, 64, head_dim)
                    expected_scales = (
                        blocks.abs().amax(dim=(-1, -2)) / 127.0 + 1.0e-7
                    )
                    scaled = blocks / expected_scales[..., None, None]
                    expected = (
                        scaled
                        + 0.5 * torch.where(scaled >= 0, 1, -1)
                    ).to(torch.int8).view_as(actual)

                    torch.testing.assert_close(
                        scales, expected_scales, rtol=2.0e-6, atol=1.0e-7
                    )
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(
                        torch.count_nonzero(actual[:, :, prefix_tokens:]).item(),
                        0,
                    )

    def test_native_anchor_route_matches_eager_active_contract(self) -> None:
        from anemoi.layers.attention.mpa.backends.sm89_k64 import (
            sm89_h3_route_precision,
        )
        from anemoi.layers.attention.mpa.routing import route_probability

        torch.manual_seed(7)
        probability = torch.rand((1, 2, 8, 8), device="cuda", dtype=torch.float16)
        anchors = torch.eye(8, device="cuda", dtype=torch.bool)
        anchor_ids = anchors.flatten().nonzero().flatten().to(torch.int32)
        eager = route_probability(
            probability,
            anchors,
            anchor_count=8,
            prefix_blocks=0,
            sparsity_ratio=0.5,
            fp8_ratio=0.75,
            fp16_ratio=0.25,
        )
        native_ids, native_low, native_fp8, native_fp16 = sm89_h3_route_precision(
            probability,
            n16=eager.fp16_blocks_per_head,
            n8=eager.fp8_blocks_per_head,
            anchors=anchors,
            anchor_ids=anchor_ids,
            anchor_count=8,
        )

        self.assertTrue(torch.equal(native_low, eager.nvfp4_counts))
        self.assertTrue(torch.equal(native_fp8, eager.fp8_counts))
        self.assertTrue(torch.equal(native_fp16, eager.fp16_counts))
        active = native_fp8 + native_fp16
        for batch in range(native_ids.size(0)):
            for head in range(native_ids.size(1)):
                for row in range(native_ids.size(2)):
                    count = int(active[batch, head, row])
                    self.assertTrue(
                        torch.equal(
                            native_ids[batch, head, row, :count],
                            eager.block_ids[batch, head, row, :count],
                        )
                    )

    def test_native_anchor_route_matches_eager_for_pure_int8(self) -> None:
        from anemoi.layers.attention.mpa.backends.sm89_k64 import (
            sm89_h3_route_precision,
        )
        from anemoi.layers.attention.mpa.routing import route_probability

        torch.manual_seed(11)
        probability = torch.rand((1, 2, 8, 8), device="cuda", dtype=torch.float16)
        anchors = torch.eye(8, device="cuda", dtype=torch.bool)
        anchor_ids = anchors.flatten().nonzero().flatten().to(torch.int32)
        eager = route_probability(
            probability,
            anchors,
            anchor_count=8,
            prefix_blocks=0,
            sparsity_ratio=0.5,
            fp8_ratio=1.0,
            fp16_ratio=0.0,
        )
        native_ids, native_low, native_int8, native_fp16 = (
            sm89_h3_route_precision(
                probability,
                n16=0,
                n8=eager.fp8_blocks_per_head,
                anchors=anchors,
                anchor_ids=anchor_ids,
                anchor_count=8,
            )
        )

        self.assertTrue(torch.equal(native_low, eager.nvfp4_counts))
        self.assertTrue(torch.equal(native_int8, eager.fp8_counts))
        self.assertTrue(torch.equal(native_fp16, eager.fp16_counts))
        for batch in range(native_ids.size(0)):
            for head in range(native_ids.size(1)):
                for row in range(native_ids.size(2)):
                    count = int(native_int8[batch, head, row])
                    self.assertTrue(
                        torch.equal(
                            native_ids[batch, head, row, :count],
                            eager.block_ids[batch, head, row, :count],
                        )
                    )

    def test_pure_int8_specialization_is_bitwise_equal_to_zero_fp16_mix(self) -> None:
        from anemoi.layers.attention.mpa.backends.sm89_k64 import (
            native_k64_mixed_attention,
            prepare_k64_fp8_operands,
        )

        torch.manual_seed(23)
        query = torch.randn((1, 2, 128, 128), device="cuda", dtype=torch.float16)
        key = torch.randn((1, 2, 192, 128), device="cuda", dtype=torch.float16)
        value = torch.randn((1, 2, 192, 128), device="cuda", dtype=torch.float16)
        operands = prepare_k64_fp8_operands(query, key, value)
        route = (
            torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
            .view(1, 1, 1, 3)
            .expand(1, 2, 2, 3)
            .contiguous()
        )
        counts = torch.full((1, 2, 2), 3, device="cuda", dtype=torch.int32)
        zeros = torch.zeros_like(counts)
        valid = torch.tensor([[64, 64, 37]], device="cuda", dtype=torch.int32)

        pure, _ = native_k64_mixed_attention(
            query,
            key,
            value,
            route,
            counts,
            route,
            zeros,
            valid,
            prepared_operands=operands,
            active_fp16=False,
        )
        reference, _ = native_k64_mixed_attention(
            query,
            key,
            value,
            route,
            counts,
            route,
            zeros,
            valid,
            prepared_operands=operands,
            active_fp16=True,
        )

        self.assertTrue(torch.isfinite(pure).all())
        self.assertTrue(torch.equal(pure, reference))


if __name__ == "__main__":
    unittest.main()
