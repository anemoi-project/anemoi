import math
import unittest
from unittest.mock import Mock, patch

import torch


class SM120Q128BackendTests(unittest.TestCase):
    def test_prefix_int8_wrapper_reuses_shared_k_and_v_operands(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q128

        operation = Mock(return_value="prefix-output")
        prefix_q8 = torch.empty((1, 14, 128, 128), dtype=torch.int8)
        prefix_q_scale = torch.empty((1, 14, 1), dtype=torch.float32)
        shared = tuple(torch.tensor(index) for index in range(6))
        valid = torch.full((1, 2), 64, dtype=torch.int32)
        with patch.object(sm120_q128, "_load_prefix_int8", return_value=operation):
            result = sm120_q128.sm120_q128_prefix_int8_attention(
                prefix_q8, prefix_q_scale, shared, valid, 95
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
                95,
                1.0 / math.sqrt(128),
            ),
        )

    def test_mxfp8_wrapper_selects_the_active_only_specialization(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q128

        operation = Mock(return_value=("output", "lse"))
        q16 = torch.empty((1, 14, 128, 128), dtype=torch.float16)
        k16 = v16 = torch.empty((1, 14, 128, 128), dtype=torch.float16)
        q8 = torch.empty_like(q16, dtype=torch.uint8)
        qs = torch.empty((1, 14, 128, 4), dtype=torch.uint8)
        k8 = torch.empty_like(k16, dtype=torch.uint8)
        ks = torch.empty((1, 14, 128, 4), dtype=torch.uint8)
        v8 = torch.empty((1, 14, 128, 128), dtype=torch.uint8)
        vs = torch.empty((1, 14, 2, 256), dtype=torch.uint8)
        ids = torch.zeros((1, 14, 1, 2), dtype=torch.int32)
        counts = torch.ones((1, 14, 1), dtype=torch.int32)
        empty = torch.empty(0, dtype=torch.int32)
        valid = torch.full((1, 2), 64, dtype=torch.int32)
        operands = (q8, qs, k8, ks, v8, vs)

        with patch.object(
            sm120_q128, "_load_mxfp8", return_value=operation
        ):
            result = sm120_q128.sm120_q128_mxfp8_attention(
                operands,
                q16,
                k16,
                v16,
                ids,
                counts,
                empty,
                valid,
                fp16_prefix_blocks=0,
                active_fp16=False,
            )

        self.assertEqual(result, ("output", "lse"))
        self.assertEqual(operation.call_args.args[:6], operands)
        self.assertEqual(operation.call_args.args[-1], False)
        self.assertAlmostEqual(operation.call_args.args[-2], 1.0 / math.sqrt(128))

    def test_nv_mxfp8_wrapper_forwards_one_existing_operator(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q128

        operation = Mock(return_value=("output", "lse"))
        nv = tuple(torch.tensor(index) for index in range(6))
        mx = tuple(torch.tensor(index) for index in range(6, 12))
        q16 = torch.empty((1, 1, 128, 128), dtype=torch.float16)
        k16 = v16 = torch.empty((1, 1, 64, 128), dtype=torch.float16)
        ids = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
        counts = torch.ones((1, 1, 1), dtype=torch.int32)
        valid = torch.full((1, 1), 64, dtype=torch.int32)
        scales = tuple(torch.tensor(index, dtype=torch.float32) for index in range(3))
        with patch.object(sm120_q128, "_load_nv_mxfp8", return_value=operation):
            result = sm120_q128.sm120_q128_nv_mxfp8_fp16_attention(
                nv, mx, q16, k16, v16, ids, counts, counts, counts, valid,
                scales, fp16_prefix_blocks=1, active_fp16=True,
            )

        self.assertEqual(result, ("output", "lse"))
        self.assertEqual(operation.call_args.args[-3], 1)
        self.assertAlmostEqual(operation.call_args.args[-2], 1.0 / math.sqrt(128))
        self.assertEqual(operation.call_args.args[-1], True)


if __name__ == "__main__":
    unittest.main()
