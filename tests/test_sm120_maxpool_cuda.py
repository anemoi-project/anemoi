import math
import os
import unittest
from pathlib import Path

import torch

from anemoi.layers.attention.mpa.backends.sm120_q64 import (
    prepare_h3_sm120_operands,
    sm120_h3_draft_probability,
    sm120_h3_route_precision,
)
from anemoi.models.minimax_h3.native_k64_attention import (
    materialize_ragged_2d_layout,
)


_CAPTURE = Path(os.environ.get("ANEMOI_SM120_MAXPOOL_CAPTURE", ""))


@unittest.skipUnless(
    _CAPTURE.is_file()
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 0),
    "requires the indexed Dense capture and SM120",
)
class SM120MaxPoolCudaTests(unittest.TestCase):
    @torch.inference_mode()
    def test_native_ragged_pool_probability_fusion_and_route(self) -> None:
        payload = torch.load(
            _CAPTURE, map_location="cpu", weights_only=True, mmap=True
        )
        prefix_tokens = 1124
        raw = {
            name: payload[name][:, :, 0].unsqueeze(1).to("cuda")
            for name in ("q", "k", "v")
        }

        def prepare(
            query: torch.Tensor, key: torch.Tensor, query_block_size: int
        ) -> tuple[torch.Tensor, ...]:
            layout = materialize_ragged_2d_layout(
                query.device,
                frames=72,
                height=24,
                width=42,
                logical_block=query_block_size,
                enable_anchors=False,
            )
            prepared = prepare_h3_sm120_operands(
                query,
                key,
                raw["v"],
                layout.indices,
                layout.slot_valid,
                layout.counts,
                prefix_tokens=prefix_tokens,
                query_block_size=query_block_size,
                has_nvfp4=False,
                has_int8=True,
                has_mxfp8=False,
                has_fp16=False,
                has_prefix_query_int8=False,
                has_maxpool=True,
                global_scales=None,
            )
            valid = layout.slot_valid.view(-1, query_block_size)
            counts = layout.counts.float().view(1, 1, -1, 1)
            for source, actual_mean, actual_max in (
                (query, prepared[0], prepared[25]),
                (key, prepared[1], prepared[26]),
            ):
                blocks = source[
                    :, :, prefix_tokens + layout.indices, :
                ].half().view(1, 1, -1, query_block_size, 128)
                mask = valid.view(1, 1, -1, query_block_size, 1)
                mean = (
                    blocks.masked_fill(~mask, 0)
                    .float()
                    .sum(3)
                    .div(counts)
                    .half()
                )
                maximum = blocks.masked_fill(~mask, -torch.inf).amax(3)
                torch.testing.assert_close(actual_mean, mean, atol=4e-3, rtol=0)
                torch.testing.assert_close(actual_max, maximum, atol=0, rtol=0)
            return prepared

        for query_block_size in (64, 128):
            transforms = (
                (lambda tensor: tensor),
                (lambda tensor: tensor.abs()),
                (lambda tensor: -tensor.abs()),
            ) if query_block_size == 64 else (lambda tensor: -tensor.abs(),)
            for transform in transforms:
                prepare(
                    transform(raw["q"]),
                    transform(raw["k"]),
                    query_block_size,
                )

        prepared = prepare(raw["q"], raw["k"], 64)
        q_mean, k_mean = prepared[0][:, :, :8], prepared[1][:, :, :8]
        q_max, k_max = prepared[25][:, :, :8], prepared[26][:, :, :8]

        def reference(q_pool: torch.Tensor, k_pool: torch.Tensor) -> torch.Tensor:
            logits = torch.baddbmm(
                torch.zeros((1, 8, 8), device="cuda", dtype=torch.float16),
                q_pool.flatten(0, 1),
                k_pool.flatten(0, 1).transpose(-1, -2),
                beta=0,
                alpha=1 / math.sqrt(128),
            )
            return logits.float().softmax(-1).half().view(1, 1, 8, 8)

        mean_probability = reference(q_mean, k_mean)
        max_probability = reference(q_max, k_max)
        for weight in (0.0, 0.1, 0.5, 1.0):
            with self.subTest(weight=weight):
                actual = sm120_h3_draft_probability(
                    q_mean, k_mean, q_max, k_max, maxpool_weight=weight
                )
                expected = (
                    (1.0 - weight) * mean_probability.float()
                    + weight * max_probability.float()
                ).half()
                torch.testing.assert_close(actual, expected, atol=5e-4, rtol=0)

        fused = sm120_h3_draft_probability(
            q_mean, k_mean, q_max, k_max, maxpool_weight=0.5
        )
        first = sm120_h3_route_precision(fused, 16, 16, 16)
        second = sm120_h3_route_precision(fused, 16, 16, 16)
        for first_tensor, second_tensor in zip(first, second, strict=True):
            self.assertTrue(torch.equal(first_tensor, second_tensor))


if __name__ == "__main__":
    unittest.main()
