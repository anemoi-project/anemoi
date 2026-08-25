import math
import unittest
from unittest.mock import Mock, patch

import torch


class SM120Q64BackendTests(unittest.TestCase):
    def test_prefix_int8_wrapper_reuses_shared_k_and_v_operands(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q64

        operation = Mock(return_value="prefix-output")
        prefix_q8 = torch.empty((1, 14, 64, 128), dtype=torch.int8)
        prefix_q_scale = torch.empty((1, 14, 1), dtype=torch.float32)
        shared = tuple(torch.tensor(index) for index in range(6))
        valid = torch.full((1, 2), 64, dtype=torch.int32)
        with patch.object(sm120_q64, "_load_prefix_int8", return_value=operation):
            result = sm120_q64.sm120_q64_prefix_int8_attention(
                prefix_q8, prefix_q_scale, shared, valid, 61
            )

        self.assertEqual(result, "prefix-output")
        self.assertEqual(
            operation.call_args.args,
            (
                prefix_q8,
                shared[2],
                shared[4],
                prefix_q_scale,
                shared[3],
                shared[5],
                valid,
                61,
                1.0 / math.sqrt(128),
            ),
        )

    def test_h3_route_wrapper_forwards_optional_anchors(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q64

        operation = Mock(return_value=("ids", "nv", "middle", "high"))
        probability = torch.empty((1, 14, 32, 32), dtype=torch.float16)
        anchors = torch.eye(32, dtype=torch.bool)
        anchor_ids = anchors.flatten().nonzero().flatten().to(torch.int32)
        with patch.object(
            sm120_q64, "_load_h3_route", return_value=operation
        ):
            plain = sm120_q64.sm120_h3_route_precision(
                probability, n16=3, n8=5, n4=7
            )
            anchored = sm120_q64.sm120_h3_route_precision(
                probability,
                n16=3,
                n8=5,
                n4=7,
                anchors=anchors,
                anchor_ids=anchor_ids,
                anchor_count=32,
            )
        self.assertEqual(plain, ("ids", "nv", "middle", "high"))
        self.assertEqual(anchored, plain)
        plain_args = operation.call_args_list[0].args
        anchor_args = operation.call_args_list[1].args
        self.assertIs(plain_args[0], probability)
        self.assertEqual(plain_args[1:], (3, 5, 7, None, None, 0))
        self.assertIs(anchor_args[0], probability)
        self.assertEqual(anchor_args[1:4], (3, 5, 7))
        self.assertIs(anchor_args[4], anchors)
        self.assertIs(anchor_args[5], anchor_ids)
        self.assertEqual(anchor_args[6], 32)

    def test_h3_route_materializer_forwards_geometry_and_prefix_phase(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q64

        operation = Mock(return_value=("ids", "nv", "middle", "high"))
        logical = torch.empty((1, 14, 32, 32), dtype=torch.int32)
        counts = torch.empty((1, 14, 32), dtype=torch.int32)
        with patch.object(
            sm120_q64, "_load_h3_route_materialization", return_value=operation
        ):
            result = sm120_q64.sm120_h3_materialize_route(
                logical,
                counts,
                counts,
                counts,
                query_block_size=128,
                prefix_blocks=2,
                prefix_phase=1,
                prefix_first=False,
                has_fp16=True,
            )
        self.assertEqual(result, ("ids", "nv", "middle", "high"))
        self.assertEqual(
            operation.call_args.args,
            (logical, counts, counts, counts, 128, 2, 1, False, True),
        )

    def test_h3_draft_wrapper_forwards_the_two_pools(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q64

        operation = Mock(return_value="probability")
        q_pool = torch.empty((1, 14, 32, 128), dtype=torch.float16)
        k_pool = torch.empty_like(q_pool)
        with patch.object(
            sm120_q64, "_load_h3_draft", return_value=operation
        ):
            result = sm120_q64.sm120_h3_draft_probability(q_pool, k_pool)
        self.assertEqual(result, "probability")
        self.assertEqual(operation.call_args.args, (q_pool, k_pool))

    def test_h3_donor_first_wrapper_passes_ragged_geometry_and_optional_scales(
        self,
    ) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q64

        operation = Mock(return_value=tuple(range(25)))
        query = key = value = torch.empty(
            (1, 2, 192, 128), dtype=torch.bfloat16
        )
        indices = torch.arange(128, dtype=torch.int64)
        valid = torch.ones(128, dtype=torch.bool)
        counts = torch.tensor([128], dtype=torch.int32)
        scales = tuple(torch.ones(1, dtype=torch.float32) for _ in range(3))

        with patch.object(
            sm120_q64, "_load_h3_preparation", return_value=operation
        ):
            result = sm120_q64.prepare_h3_sm120_operands(
                query,
                key,
                value,
                indices,
                valid,
                counts,
                prefix_tokens=64,
                query_block_size=128,
                has_nvfp4=True,
                has_int8=True,
                has_mxfp8=True,
                has_fp16=True,
                has_prefix_query_int8=True,
                global_scales=scales,
            )

        self.assertEqual(result, tuple(range(25)))
        self.assertEqual(
            operation.call_args.args,
            (
                query,
                key,
                value,
                indices,
                valid,
                counts,
                64,
                128,
                True,
                True,
                True,
                True,
                True,
                *scales,
            ),
        )

    def test_wrapper_passes_the_exact_fp16_contract(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q64

        operation = Mock(return_value=("output", "lse"))
        query = torch.empty((1, 14, 64, 128), dtype=torch.float16)
        key = value = torch.empty((1, 14, 128, 128), dtype=torch.float16)
        ids = torch.zeros((1, 14, 1, 2), dtype=torch.int32)
        counts = torch.ones((1, 14, 1), dtype=torch.int32)
        valid = torch.full((1, 2), 64, dtype=torch.int32)
        with patch.object(sm120_q64, "_load_ops", return_value=operation):
            result = sm120_q64.sm120_q64_fp16_attention(
                query, key, value, ids, counts, valid
            )
        self.assertEqual(result, ("output", "lse"))
        self.assertAlmostEqual(
            operation.call_args.args[-1], 1.0 / math.sqrt(128)
        )

    def test_wrapper_passes_the_exact_mxfp8_contract(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q64

        operation = Mock(return_value=("output", "lse"))
        q16 = torch.empty((1, 14, 64, 128), dtype=torch.float16)
        k16 = v16 = torch.empty((1, 14, 128, 128), dtype=torch.float16)
        q8 = torch.empty_like(q16, dtype=torch.uint8)
        qs = torch.empty((1, 14, 64, 4), dtype=torch.uint8)
        k8 = torch.empty_like(k16, dtype=torch.uint8)
        ks = torch.empty((1, 14, 128, 4), dtype=torch.uint8)
        v8 = torch.empty((1, 14, 128, 128), dtype=torch.uint8)
        vs = torch.empty((1, 14, 2, 256), dtype=torch.uint8)
        ids = torch.zeros((1, 14, 1, 2), dtype=torch.int32)
        low = high = torch.ones((1, 14, 1), dtype=torch.int32)
        valid = torch.full((1, 2), 64, dtype=torch.int32)
        operands = (q8, qs, k8, ks, v8, vs, q16, k16, v16)
        with patch.object(
            sm120_q64, "_load_mxfp8_op", return_value=operation
        ):
            result = sm120_q64.sm120_q64_mxfp8_attention(
                *operands,
                ids,
                low,
                high,
                valid,
                fp16_prefix_blocks=1,
            )
        self.assertEqual(result, ("output", "lse"))
        self.assertEqual(len(operation.call_args.args), 16)
        self.assertEqual(operation.call_args.args[-3], 1)
        self.assertAlmostEqual(
            operation.call_args.args[-2], 1.0 / math.sqrt(128)
        )
        self.assertEqual(operation.call_args.args[-1], True)

    def test_mxfp8_wrapper_can_select_the_active_only_specialization(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q64

        operation = Mock(return_value=("output", "lse"))
        q16 = torch.empty((1, 1, 64, 128), dtype=torch.float16)
        k16 = v16 = torch.empty((1, 1, 64, 128), dtype=torch.float16)
        q8 = torch.empty_like(q16, dtype=torch.uint8)
        qs = ks = torch.empty((1, 1, 64, 4), dtype=torch.uint8)
        k8 = torch.empty_like(k16, dtype=torch.uint8)
        v8 = torch.empty((1, 1, 128, 64), dtype=torch.uint8)
        vs = torch.empty((1, 1, 1, 256), dtype=torch.uint8)
        ids = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
        counts = torch.ones((1, 1, 1), dtype=torch.int32)
        empty = torch.empty(0, dtype=torch.int32)
        valid = torch.full((1, 1), 64, dtype=torch.int32)
        with patch.object(sm120_q64, "_load_mxfp8_op", return_value=operation):
            result = sm120_q64.sm120_q64_mxfp8_attention(
                q8, qs, k8, ks, v8, vs, q16, k16, v16,
                ids, counts, empty, valid, active_fp16=False,
            )

        self.assertEqual(result, ("output", "lse"))
        self.assertEqual(operation.call_args.args[-3], 0)
        self.assertAlmostEqual(operation.call_args.args[-2], 1.0 / math.sqrt(128))
        self.assertEqual(operation.call_args.args[-1], False)


if __name__ == "__main__":
    unittest.main()
