from __future__ import annotations

import inspect
import unittest
from unittest.mock import Mock, patch

import torch

from anemoi.models.minimax_h3 import native_k64_attention as native
from anemoi.models.minimax_h3 import mpa_attention as adapter
from anemoi.models.minimax_h3.mpa_attention import H3MPAAttention, H3MPAConfig

LEGAL = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.6, 0.4, 0.0, 0.0),
    (0.6, 0.0, 0.4, 0.0),
    (0.85, 0.0, 0.0, 0.15),
    (0.0, 0.85, 0.0, 0.15),
    (0.0, 0.0, 0.85, 0.15),
    (0.6, 0.25, 0.0, 0.15),
    (0.6, 0.0, 0.25, 0.15),
)

ACCEPTED = {
    64: {LEGAL[index] for index in (0, 1, 2, 3, 8, 10)},
    128: {LEGAL[index] for index in (0, 1, 2, 3, 4, 5, 7, 8, 10)},
}


class MiniMaxH3PhaseDispatchTests(unittest.TestCase):
    def test_adapter_forwards_explicit_prefix_precision_to_sm120(self) -> None:
        attention = H3MPAAttention(
            H3MPAConfig(
                video_shape=(1, 1, 128),
                prefix_tokens=64,
                query_block_size=128,
                prefix_kv_precision="mxfp8",
                prefix_query_precision="int8",
                fp8_ratio=0.0,
                nvfp4_ratio=0.6,
                mxfp8_ratio=0.4,
                fp16_ratio=0.0,
            )
        )
        q = torch.empty((192, 1, 128), dtype=torch.float16)

        with patch.object(attention, "_validate_qkv"), patch.object(
            adapter, "_can_use_direct_packed_views", return_value=False
        ), patch.object(
            adapter, "sm120_ragged_h3_attention", return_value=q.unsqueeze(0)
        ) as operation:
            attention.mpa(q, q, q, layer=0)

        self.assertEqual(
            operation.call_args.kwargs["prefix_kv_precision"], "mxfp8"
        )
        self.assertEqual(
            operation.call_args.kwargs["prefix_query_precision"], "int8"
        )

    def test_sm120_entry_prefix_precision_defaults_to_auto(self) -> None:
        parameter = inspect.signature(
            native.sm120_ragged_h3_attention
        ).parameters["prefix_kv_precision"]

        self.assertEqual(parameter.default, "auto")
        query_parameter = inspect.signature(
            native.sm120_ragged_h3_attention
        ).parameters["prefix_query_precision"]
        self.assertEqual(query_parameter.default, "auto")

    def test_sm120_profile_ranges_are_opt_in_and_cover_complete_components(self) -> None:
        signature = inspect.signature(native.sm120_ragged_h3_attention)
        source = inspect.getsource(native.sm120_ragged_h3_attention)

        self.assertFalse(signature.parameters["profile_nvtx"].default)
        for label in (
            "anemoi.complete",
            "anemoi.dense_prefix",
            "anemoi.peripheral.prepare",
            "anemoi.peripheral.draft_route_materialize",
            "anemoi.mpa_mainloop",
            "anemoi.peripheral.assembly",
        ):
            self.assertIn(label, source)

    def test_prefix_phase_resolves_auto_and_explicit_precision(self) -> None:
        video_phases = ("nvfp4", "mxfp8")

        self.assertEqual(
            native._resolve_sm120_prefix_phase(video_phases, "auto"),
            "nvfp4",
        )
        self.assertEqual(
            native._resolve_sm120_prefix_phase(video_phases, "fp16"),
            "fp16",
        )
        self.assertEqual(
            native._active_sm120_phases(("nvfp4",), "mxfp8"),
            ("nvfp4", "mxfp8"),
        )
        with self.assertRaisesRegex(ValueError, "prefix_kv_precision"):
            native._resolve_sm120_prefix_phase(video_phases, "fp8")

    def test_prefix_query_precision_resolves_from_the_middle_phase(self) -> None:
        self.assertEqual(
            native._resolve_sm120_prefix_query_precision(("int8",), "auto"),
            "int8",
        )
        self.assertEqual(
            native._resolve_sm120_prefix_query_precision(
                ("nvfp4", "int8"), "auto"
            ),
            "int8",
        )
        self.assertEqual(
            native._resolve_sm120_prefix_query_precision(("mxfp8",), "auto"),
            "fp16",
        )
        self.assertEqual(
            native._resolve_sm120_prefix_query_precision(("int8",), "mxfp8"),
            "mxfp8",
        )
        with self.assertRaisesRegex(ValueError, "prefix_query_precision"):
            native._resolve_sm120_prefix_query_precision(("int8",), "fp8")

    def test_q64_prefix_blocks_are_inserted_at_the_selected_phase(self) -> None:
        logical = torch.tensor([[[[2, 0, 1, 3]]]], dtype=torch.int32)
        counts = torch.ones((1, 1, 1), dtype=torch.int32)
        high_counts = torch.full((1, 1, 1), 2, dtype=torch.int32)
        expected = {
            "nvfp4": ([0, 1, 4, 2, 3, 5], (3, 1, 2)),
            "mxfp8": ([4, 0, 1, 2, 3, 5], (1, 3, 2)),
            "fp16": ([4, 2, 3, 5, 0, 1], (1, 1, 4)),
        }

        for prefix_phase, (ids_expected, counts_expected) in expected.items():
            with self.subTest(prefix_phase=prefix_phase):
                ids, nv, middle, high = native._compose_sm120_phase_route(
                    logical,
                    counts,
                    counts,
                    high_counts,
                    query_block_size=64,
                    prefix_blocks=2,
                    prefix_phase=prefix_phase,
                    active_phases=("nvfp4", "mxfp8", "fp16"),
                )

                self.assertEqual(ids.flatten().tolist(), ids_expected)
                self.assertEqual(
                    (nv.item(), middle.item(), high.item()), counts_expected
                )

    def test_q128_prefix_precision_preserves_physical_video_stages(self) -> None:
        logical = torch.tensor([[[[2, 0, 1, 3]]]], dtype=torch.int32)
        counts = torch.ones((1, 1, 1), dtype=torch.int32)
        high_counts = torch.full((1, 1, 1), 2, dtype=torch.int32)
        video_stages = []

        for prefix_phase in ("nvfp4", "mxfp8", "fp16"):
            ids, _, _, _ = native._compose_sm120_phase_route(
                logical,
                counts,
                counts,
                high_counts,
                query_block_size=128,
                prefix_blocks=2,
                prefix_phase=prefix_phase,
                active_phases=("nvfp4", "mxfp8", "fp16"),
            )
            video_stages.append([value for value in ids.flatten().tolist() if value >= 2])

        self.assertEqual(
            video_stages,
            [[6, 7, 2, 3, 4, 5, 8, 9]] * 3,
        )

    def test_auto_fp16_prefix_keeps_the_legacy_concat_fast_path(self) -> None:
        logical = torch.tensor([[[[2, 0, 1, 3]]]], dtype=torch.int32)
        counts = torch.ones((1, 1, 1), dtype=torch.int32)

        with patch.object(native.torch, "empty", side_effect=AssertionError):
            ids, _, _, high = native._compose_sm120_phase_route(
                logical,
                counts,
                counts,
                counts,
                query_block_size=64,
                prefix_blocks=2,
                prefix_phase="fp16",
                active_phases=("nvfp4", "mxfp8", "fp16"),
                legacy_auto_prefix_order=True,
            )

        self.assertEqual(ids.flatten().tolist(), [4, 2, 3, 5, 0, 1])
        self.assertEqual(high.item(), 3)

    def test_explicit_fp16_prefix_precedes_unused_route_capacity(self) -> None:
        logical = torch.tensor([[[[2, 0, 1, 3]]]], dtype=torch.int32)
        one = torch.ones((1, 1, 1), dtype=torch.int32)
        zero = torch.zeros_like(one)

        ids, _, _, high = native._compose_sm120_phase_route(
            logical,
            one,
            zero,
            one,
            query_block_size=64,
            prefix_blocks=2,
            prefix_phase="fp16",
            active_phases=("nvfp4", "fp16"),
        )

        self.assertEqual(ids.flatten().tolist(), [4, 2, 0, 1, 3, 5])
        self.assertEqual(high.item(), 3)

    def test_zero_route_tail_stays_outside_active_composed_counts(self) -> None:
        logical = torch.tensor([[[[2, 0, 0, 0]]]], dtype=torch.int32)
        one = torch.ones((1, 1, 1), dtype=torch.int32)
        zero = torch.zeros_like(one)

        ids, nv, middle, high = native._compose_sm120_phase_route(
            logical,
            one,
            zero,
            zero,
            query_block_size=64,
            prefix_blocks=2,
            prefix_phase="fp16",
            active_phases=("nvfp4", "fp16"),
        )

        active = nv.item() + middle.item() + high.item()
        self.assertEqual(ids.flatten()[:active].tolist(), [4, 0, 1])
        self.assertEqual(active, 3)

    def test_sm120_route_uses_native_path_without_anchors(self) -> None:
        probability = torch.empty((1, 2, 4, 4), dtype=torch.float16)
        routed = (
            torch.zeros((1, 2, 4, 4), dtype=torch.int32),
            torch.ones((1, 2, 4), dtype=torch.int32),
            torch.full((1, 2, 4), 2, dtype=torch.int32),
            torch.full((1, 2, 4), 3, dtype=torch.int32),
        )
        with patch.object(
            native, "_route_counts", return_value=(3, 5, 7)
        ), patch.object(
            native, "sm120_h3_route_precision", return_value=routed
        ) as operation, patch.object(
            native, "route_probability", side_effect=AssertionError
        ):
            route = native._route_sm120_probability(
                probability,
                None,
                None,
                anchor_count=0,
                sparsity_ratio=0.5,
                ratios=(0.4, 0.6, 0.0, 0.0),
            )

        self.assertIs(operation.call_args.args[0], probability)
        self.assertEqual(operation.call_args.args[1:], (3, 5, 7, None, None, 0))
        self.assertIs(route.block_ids, routed[0])
        self.assertEqual(
            (
                route.fp16_blocks_per_head,
                route.fp8_blocks_per_head,
                route.nvfp4_blocks_per_head,
            ),
            (3, 5, 7),
        )

    def test_sm120_route_uses_native_path_with_anchors(self) -> None:
        probability = torch.empty((1, 2, 4, 4), dtype=torch.float16)
        anchors = torch.eye(4, dtype=torch.bool)
        anchor_ids = anchors.flatten().nonzero().flatten().to(torch.int32)
        routed = (
            torch.zeros((1, 2, 4, 4), dtype=torch.int32),
            torch.ones((1, 2, 4), dtype=torch.int32),
            torch.full((1, 2, 4), 2, dtype=torch.int32),
            torch.full((1, 2, 4), 3, dtype=torch.int32),
        )
        with patch.object(
            native, "_route_counts", return_value=(3, 5, 7)
        ), patch.object(
            native, "sm120_h3_route_precision", return_value=routed
        ) as operation, patch.object(
            native, "route_probability", side_effect=AssertionError
        ):
            route = native._route_sm120_probability(
                probability,
                anchors,
                anchor_ids,
                anchor_count=4,
                sparsity_ratio=0.5,
                ratios=(0.0, 0.85, 0.0, 0.15),
            )

        self.assertIs(route.block_ids, routed[0])
        self.assertIs(operation.call_args.args[0], probability)
        self.assertEqual(operation.call_args.args[1:4], (3, 5, 7))
        self.assertIs(operation.call_args.args[4], anchors)
        self.assertIs(operation.call_args.args[5], anchor_ids)
        self.assertEqual(operation.call_args.args[6], 4)

    def test_production_sm120_path_selects_the_donor_first_boundary(self) -> None:
        source = inspect.getsource(native.sm120_ragged_h3_attention)
        self.assertIn("_prepare_h3_sm120_inputs(", source)
        self.assertIn("prepared_nv_operands=nv_operands", source)
        self.assertIn("prepared_int8_operands=int8_operands", source)
        self.assertIn("prepared_mxfp8_operands=mxfp8_operands", source)
        self.assertIn(
            "prefix_phase = _resolve_sm120_prefix_phase(",
            source,
        )
        self.assertIn("prefix_phase=prefix_phase", source)
        self.assertIn("sm120_h3_draft_probability(q_pool, k_pool)", source)
        self.assertIn("_route_sm120_probability(", source)
        self.assertEqual(source.count("sm120_h3_materialize_route("), 1)
        self.assertNotIn("if layout.anchors is None", source)
        self.assertNotIn('prefix_kv_precision != "auto"', source)
        self.assertIn("if diag_jensen", source)

    def test_sm120_overlaps_dense_prefix_with_preparation(self) -> None:
        source = inspect.getsource(native.sm120_ragged_h3_attention)
        self.assertIn("prefix_stream.wait_stream(current_stream)", source)
        self.assertIn("current_stream.wait_stream(prefix_stream)", source)

    def test_sm120_prefix_query_dispatch_preserves_both_stream_schedules(
        self,
    ) -> None:
        source = inspect.getsource(native.sm120_ragged_h3_attention)
        self.assertIn(
            "resolved_prefix_query_precision = "
            "_resolve_sm120_prefix_query_precision(",
            source,
        )
        self.assertIn('resolved_prefix_query_precision == "int8"', source)
        self.assertIn('and "int8" in video_phases', source)
        self.assertIn("has_prefix_query_int8=native_prefix_int8", source)
        self.assertIn("sm120_q64_prefix_int8_attention", source)
        self.assertIn("sm120_q128_prefix_int8_attention", source)
        self.assertEqual(source.count("current_stream.wait_stream(prefix_stream)"), 1)

        fallback = source.index("if not native_prefix_int8:")
        first_wait = source.index("prefix_stream.wait_stream(current_stream)")
        sdpa = source.index("_dense_prefix_sdpa(", first_wait)
        preparation = source.index("_prepare_h3_sm120_inputs(")
        valid_k = source.index("valid_k =", preparation)
        second_wait = source.index(
            "prefix_stream.wait_stream(current_stream)", first_wait + 1
        )
        prefix_launch = source.index("prefix_function(", second_wait)
        draft = source.index("sm120_h3_draft_probability", prefix_launch)
        self.assertLess(fallback, first_wait)
        self.assertLess(sdpa, preparation)
        self.assertLess(preparation, valid_k)
        self.assertLess(valid_k, second_wait)
        self.assertLess(second_wait, prefix_launch)
        self.assertLess(prefix_launch, draft)
        self.assertNotIn(".permute(", source[second_wait:draft])

    def test_donor_first_boundary_uses_native_pools_and_active_formats(
        self,
    ) -> None:
        raw = torch.empty((1, 2, 192, 128), dtype=torch.bfloat16)
        indices = torch.arange(128, dtype=torch.int64)
        valid = torch.ones(128, dtype=torch.bool)
        counts = torch.tensor([128], dtype=torch.int32)
        scales = ("q-scale", "k-scale", "v-scale")
        packed_q = torch.empty((1, 2, 128, 128), dtype=torch.float16)
        packed_k = packed_v = torch.zeros(
            (1, 2, 192, 128), dtype=torch.float16
        )
        prepared = (
            "donor-q-pool",
            "donor-k-pool",
            packed_q,
            packed_k,
            packed_v,
            *(f"prepared-{index}" for index in range(20)),
        )

        with patch.object(
            native, "prepare_h3_sm120_operands", return_value=prepared
        ) as operation, patch.object(
            native,
            "_pool_query",
            side_effect=AssertionError("native pools must not be recomputed"),
        ) as pool:
            result = native._prepare_h3_sm120_inputs(
                raw,
                raw,
                raw,
                video_token_indices=indices,
                video_slot_valid=valid,
                video_valid_counts=counts,
                prefix_tokens=64,
                query_block_size=128,
                phases=("nvfp4", "int8", "fp16"),
                has_prefix_query_int8=True,
                global_scales=scales,
            )

        self.assertEqual(
            result,
            (
                "donor-q-pool",
                "donor-k-pool",
                *prepared[2:5],
                prepared[5:11],
                prepared[11:17],
                prepared[17:23],
                prepared[23:25],
            ),
        )
        pool.assert_not_called()
        self.assertEqual(operation.call_args.kwargs["prefix_tokens"], 64)
        self.assertEqual(operation.call_args.kwargs["query_block_size"], 128)
        self.assertTrue(operation.call_args.kwargs["has_int8"])
        self.assertTrue(operation.call_args.kwargs["has_prefix_query_int8"])
        self.assertEqual(operation.call_args.kwargs["global_scales"], scales)

    def test_phase_runner_consumes_donor_first_operands_without_rereading_packed(
        self,
    ) -> None:
        q = torch.empty((1, 1, 128, 128), dtype=torch.float16)
        k = v = torch.empty((1, 1, 64, 128), dtype=torch.float16)
        ids = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
        counts = torch.ones((1, 1, 1), dtype=torch.int32)
        valid = torch.full((1, 1), 64, dtype=torch.int32)
        nv = tuple(f"nv-{index}" for index in range(6))
        mx = tuple(f"mx-{index}" for index in range(6))
        scales = ("q-scale", "k-scale", "v-scale")

        with patch.object(
            native,
            "sm120_q128_nv_mxfp8_fp16_attention",
            return_value=("output", "lse"),
        ) as attention, patch.object(
            native, "prepare_q128_nvfp4"
        ) as old_nv, patch.object(
            native, "prepare_mxfp8"
        ) as old_mx:
            result = native._run_sm120_phases(
                query_block_size=128,
                ratios=(0.6, 0.0, 0.25, 0.15),
                query_fp16=q,
                key_fp16=k,
                value_fp16=v,
                block_ids=ids,
                nvfp4_counts=counts,
                middle_counts=counts,
                fp16_counts=counts,
                valid_k_counts=valid,
                layer=0,
                fp16_prefix_blocks=1,
                prepared_nv_operands=nv,
                prepared_mxfp8_operands=mx,
                prepared_global_scales=scales,
            )

        self.assertEqual(result, ("output", "lse"))
        old_nv.assert_not_called()
        old_mx.assert_not_called()
        self.assertIs(attention.call_args.args[0], nv)
        self.assertIs(attention.call_args.args[1], mx)

    def test_phase_runner_consumes_prepared_int8_without_rereading_packed(
        self,
    ) -> None:
        q = torch.empty((1, 1, 128, 128), dtype=torch.float16)
        k = v = torch.empty((1, 1, 64, 128), dtype=torch.float16)
        ids = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
        counts = torch.ones((1, 1, 1), dtype=torch.int32)
        valid = torch.full((1, 1), 64, dtype=torch.int32)
        int8 = tuple(f"int8-{index}" for index in range(6))

        with patch.object(
            native,
            "sm120_q128_int8_fp16_attention",
            return_value=("output", "lse"),
        ) as attention, patch.object(
            native, "prepare_k64_fp8_operands"
        ) as old_int8:
            result = native._run_sm120_phases(
                query_block_size=128,
                ratios=(0.0, 1.0, 0.0, 0.0),
                prefix_phase="int8",
                query_fp16=q,
                key_fp16=k,
                value_fp16=v,
                block_ids=ids,
                nvfp4_counts=counts,
                middle_counts=counts,
                fp16_counts=torch.empty(0, dtype=torch.int32),
                valid_k_counts=valid,
                layer=0,
                fp16_prefix_blocks=1,
                prepared_int8_operands=int8,
            )

        self.assertEqual(result, ("output", "lse"))
        old_int8.assert_not_called()
        self.assertIs(attention.call_args.args[0], int8)

    def test_prefix_only_phases_reuse_existing_attention_families(self) -> None:
        q = torch.empty((1, 1, 128, 128), dtype=torch.float16)
        k = v = torch.empty((1, 1, 64, 128), dtype=torch.float16)
        ids = torch.zeros((1, 1, 1, 2), dtype=torch.int32)
        counts = torch.ones((1, 1, 1), dtype=torch.int32)
        valid = torch.full((1, 1), 64, dtype=torch.int32)
        cases = (
            (64, (1.0, 0.0, 0.0, 0.0), "fp16", "sm120_q64_nvfp4_fp16_attention", True),
            (64, (0.6, 0.0, 0.4, 0.0), "mxfp8", "sm120_q64_nv_mxfp8_fp16_attention", False),
            (128, (0.6, 0.0, 0.4, 0.0), "fp16", "sm120_q128_nv_mxfp8_fp16_attention", True),
            (128, (0.6, 0.4, 0.0, 0.0), "fp16", "sm120_q128_nv_int8_fp16_attention", True),
        )
        targets = {case[3] for case in cases}
        attention_mocks = {
            name: Mock(return_value=("output", "lse")) for name in targets
        }

        with patch.object(
            native, "_nvfp4_global_scales", return_value=(1, 2, 3)
        ), patch.object(
            native, "prepare_q64_nvfp4", return_value=("nv",)
        ), patch.object(
            native, "prepare_q128_nvfp4", return_value=("nv",)
        ), patch.object(
            native, "prepare_mxfp8", return_value=("mx",)
        ), patch.object(
            native, "prepare_k64_fp8_operands", return_value=tuple(range(6))
        ):
            patchers = [
                patch.object(native, name, operation)
                for name, operation in attention_mocks.items()
            ]
            try:
                for patcher in patchers:
                    patcher.start()
                for query_block, ratios, prefix, target, active_fp16 in cases:
                    with self.subTest(
                        query_block=query_block,
                        ratios=ratios,
                        prefix=prefix,
                    ):
                        before = {
                            name: operation.call_count
                            for name, operation in attention_mocks.items()
                        }
                        native._run_sm120_phases(
                            query_block_size=query_block,
                            ratios=ratios,
                            prefix_phase=prefix,
                            query_fp16=q[:, :, :query_block],
                            key_fp16=k,
                            value_fp16=v,
                            block_ids=ids,
                            nvfp4_counts=counts,
                            middle_counts=counts,
                            fp16_counts=(
                                counts
                                if active_fp16
                                else torch.empty(0, dtype=torch.int32)
                            ),
                            valid_k_counts=valid,
                            layer=0,
                            fp16_prefix_blocks=1,
                        )

                        self.assertEqual(
                            sum(
                                operation.call_count - before[name]
                                for name, operation in attention_mocks.items()
                            ),
                            1,
                        )
                        operation = attention_mocks[target]
                        self.assertEqual(
                            operation.call_args.kwargs["active_fp16"],
                            active_fp16,
                        )
                        self.assertEqual(
                            operation.call_args.kwargs["fp16_prefix_blocks"],
                            1 if prefix == "fp16" else 0,
                        )
            finally:
                for patcher in reversed(patchers):
                    patcher.stop()

    def test_all_legal_sm120_configs_parse_for_q64_and_q128(self) -> None:
        for query_block in (64, 128):
            for nvfp4, int8, mxfp8, fp16 in LEGAL:
                with self.subTest(query_block=query_block, ratios=(nvfp4, int8, mxfp8, fp16)):
                    config = H3MPAConfig(
                        query_block_size=query_block,
                        fp8_ratio=0.0,
                        nvfp4_ratio=nvfp4,
                        int8_ratio=int8,
                        mxfp8_ratio=mxfp8,
                        fp16_ratio=fp16,
                    )
                    self.assertEqual(config.precision(0), (nvfp4, int8, mxfp8, fp16))

    def test_int8_and_mxfp8_cannot_share_a_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "alternative middle phases"):
            H3MPAConfig(
                query_block_size=128,
                fp8_ratio=0.0,
                nvfp4_ratio=0.0,
                int8_ratio=0.5,
                mxfp8_ratio=0.5,
                fp16_ratio=0.0,
            )

    def test_zero_total_sm120_ratio_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum(?:ming)? to one"):
            H3MPAConfig(
                query_block_size=128,
                fp8_ratio=0.0,
                nvfp4_ratio=0.0,
                int8_ratio=0.0,
                mxfp8_ratio=0.0,
                fp16_ratio=0.0,
            )

    def test_pending_cells_stop_before_any_preparation(self) -> None:
        pending = {
            query_block: set(LEGAL) - accepted
            for query_block, accepted in ACCEPTED.items()
        }
        with patch.object(native, "prepare_q128_nvfp4") as nv_prepare, patch.object(
            native, "prepare_k64_fp8_operands"
        ) as int8_prepare, patch.object(native, "prepare_mxfp8") as mx_prepare:
            for query_block, configs in pending.items():
                for ratios in configs:
                    with self.subTest(query_block=query_block, ratios=ratios):
                        with self.assertRaisesRegex(RuntimeError, "pending"):
                            native._run_sm120_phases(
                                query_block_size=query_block,
                                ratios=ratios,
                                query_fp16=torch.empty((1, 1, query_block, 128), dtype=torch.float16),
                                key_fp16=torch.empty((1, 1, 64, 128), dtype=torch.float16),
                                value_fp16=torch.empty((1, 1, 64, 128), dtype=torch.float16),
                                block_ids=torch.zeros((1, 1, 1, 1), dtype=torch.int32),
                                nvfp4_counts=torch.zeros((1, 1, 1), dtype=torch.int32),
                                middle_counts=torch.zeros((1, 1, 1), dtype=torch.int32),
                                fp16_counts=torch.zeros((1, 1, 1), dtype=torch.int32),
                                valid_k_counts=torch.full((1, 1), 64, dtype=torch.int32),
                                layer=0,
                                fp16_prefix_blocks=0,
                            )
        nv_prepare.assert_not_called()
        int8_prepare.assert_not_called()
        mx_prepare.assert_not_called()

    def test_q64_nvfp4_int8_can_run_only_as_explicit_diagnostic(self) -> None:
        ratios = (0.6, 0.4, 0.0, 0.0)
        q = torch.empty((1, 1, 64, 128), dtype=torch.float16)
        k = v = torch.empty((1, 1, 64, 128), dtype=torch.float16)
        ids = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
        counts = torch.ones((1, 1, 1), dtype=torch.int32)
        valid = torch.full((1, 1), 64, dtype=torch.int32)
        nv = tuple(f"nv-{index}" for index in range(6))
        int8 = tuple(f"int8-{index}" for index in range(6))
        scales = ("q-scale", "k-scale", "v-scale")

        with self.assertRaisesRegex(RuntimeError, "pending"):
            native._require_accepted_sm120_phase_config(64, ratios, "int8")
        with self.assertRaisesRegex(RuntimeError, "pending"):
            native._require_accepted_sm120_phase_config(
                64, ratios, "fp16", diagnostic_only=True
            )

        with patch.object(
            native,
            "sm120_q64_nv_int8_fp16_attention",
            return_value=("output", "lse"),
        ) as attention:
            result = native._run_sm120_phases(
                query_block_size=64,
                ratios=ratios,
                prefix_phase="int8",
                query_fp16=q,
                key_fp16=k,
                value_fp16=v,
                block_ids=ids,
                nvfp4_counts=counts,
                middle_counts=counts,
                fp16_counts=torch.empty(0, dtype=torch.int32),
                valid_k_counts=valid,
                layer=0,
                fp16_prefix_blocks=1,
                prepared_nv_operands=nv,
                prepared_int8_operands=int8,
                prepared_global_scales=scales,
                diagnostic_only=True,
            )

        self.assertEqual(result, ("output", "lse"))
        self.assertIs(attention.call_args.args[0], nv)
        self.assertIs(attention.call_args.args[1], int8)

    def test_accepted_cells_prepare_only_active_formats_and_call_one_family(self) -> None:
        q = torch.empty((1, 1, 128, 128), dtype=torch.float16)
        k = v = torch.empty((1, 1, 64, 128), dtype=torch.float16)
        ids = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
        counts = torch.ones((1, 1, 1), dtype=torch.int32)
        valid = torch.full((1, 1), 64, dtype=torch.int32)
        targets = (
            "sm120_q64_fp16_attention",
            "sm120_q64_int8_fp16_attention",
            "sm120_q64_mxfp8_attention",
            "sm120_q64_nvfp4_fp16_attention",
            "sm120_q64_nv_mxfp8_fp16_attention",
            "sm120_q128_fp16_attention",
            "sm120_q128_int8_fp16_attention",
            "sm120_q128_mxfp8_attention",
            "sm120_q128_nvfp4_fp16_attention",
            "sm120_q128_nv_int8_fp16_attention",
            "sm120_q128_nv_mxfp8_fp16_attention",
        )
        patches = {name: Mock(return_value=("output", "lse")) for name in targets}
        context = [patch.object(native, name, mock) for name, mock in patches.items()]
        with patch.object(native, "_nvfp4_global_scales", return_value=(1, 2, 3)), patch.object(
            native, "prepare_q64_nvfp4", return_value=("nv",)
        ) as q64_nv, patch.object(
            native, "prepare_q128_nvfp4", return_value=("nv",)
        ) as q128_nv, patch.object(
            native, "prepare_k64_fp8_operands", return_value=tuple(range(6))
        ) as int8, patch.object(
            native, "prepare_mxfp8", return_value=("mx",)
        ) as mx:
            entered = []
            try:
                for item in context:
                    entered.append(item)
                    item.start()
                for query_block, configs in ACCEPTED.items():
                    for ratios in configs:
                        with self.subTest(query_block=query_block, ratios=ratios):
                            before = {name: mock.call_count for name, mock in patches.items()}
                            before_prep = (q64_nv.call_count, q128_nv.call_count, int8.call_count, mx.call_count)
                            native._run_sm120_phases(
                                query_block_size=query_block,
                                ratios=ratios,
                                query_fp16=q[:, :, :query_block],
                                key_fp16=k,
                                value_fp16=v,
                                block_ids=ids,
                                nvfp4_counts=counts,
                                middle_counts=counts,
                                fp16_counts=counts,
                                valid_k_counts=valid,
                                layer=0,
                                fp16_prefix_blocks=1 if ratios[3] else 0,
                            )
                            self.assertEqual(sum(mock.call_count - before[name] for name, mock in patches.items()), 1)
                            after_prep = (q64_nv.call_count, q128_nv.call_count, int8.call_count, mx.call_count)
                            delta = tuple(after - before for after, before in zip(after_prep, before_prep))
                            self.assertEqual(delta[0] + delta[1], int(ratios[0] > 0))
                            self.assertEqual(delta[2], int(ratios[1] > 0))
                            self.assertEqual(delta[3], int(ratios[2] > 0))
            finally:
                for item in reversed(entered):
                    item.stop()


if __name__ == "__main__":
    unittest.main()
