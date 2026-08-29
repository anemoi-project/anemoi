import math
import os
import unittest
from pathlib import Path

import torch

from anemoi.layers.attention.mpa.backends.sm120_q64 import (
    sm120_h3_k_tail_r1_probability,
    sm120_h3_k_tail_r2_probability,
)
from anemoi.layers.attention.mpa.ragged_2d import make_ragged_2d_partition

_CAPTURE = Path(os.environ.get("ANEMOI_SM120_K_TAIL_CAPTURE", ""))


@unittest.skipUnless(
    _CAPTURE.is_file()
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 0),
    "requires the indexed Dense capture and SM120",
)
class SM120KTailCudaTests(unittest.TestCase):
    @torch.inference_mode()
    def test_native_probability_matches_fp16_operand_reference(self) -> None:
        payload = torch.load(_CAPTURE, map_location="cpu", weights_only=True, mmap=True)
        blocks = make_ragged_2d_partition(24, 42, 64, include_adjacency=False).blocks[:4]
        counts = torch.tensor([len(block) for block in blocks], device="cuda", dtype=torch.int32)

        def pack(name: str) -> torch.Tensor:
            source = payload[name][0, :, 0]
            packed = torch.zeros((4, 64, 128), device="cuda", dtype=torch.float16)
            for block_id, block in enumerate(blocks):
                packed[block_id, : len(block)] = source[
                    torch.tensor(block, dtype=torch.int64) + 951
                ].to(device="cuda", dtype=torch.float16)
            return packed.view(1, 1, 4 * 64, 128)

        packed_q = pack("q")
        packed_k = pack("k")
        denominator = counts.float().view(1, 1, 4, 1)
        q_pool = packed_q.view(1, 1, 4, 64, 128).float().sum(3).div(denominator).half()
        k_blocks = packed_k.view(1, 1, 4, 64, 128)
        k_pool = k_blocks.float().sum(3).div(denominator).half()

        distance = (k_blocks.float() - k_pool.float().unsqueeze(3)).square().sum(-1)
        distance.masked_fill_(
            torch.arange(64, device="cuda").view(1, 1, 1, 64) >= counts.view(1, 1, 4, 1),
            -torch.inf,
        )
        key_counts = counts.float().view(1, 1, 1, 4)
        for tail_rank, operation in (
            (1, sm120_h3_k_tail_r1_probability),
            (2, sm120_h3_k_tail_r2_probability),
        ):
            with self.subTest(tail_rank=tail_rank):
                extreme_ids = distance.topk(tail_rank, dim=-1).indices
                extremes = torch.gather(
                    k_blocks,
                    3,
                    extreme_ids.unsqueeze(-1).expand(-1, -1, -1, -1, 128),
                )
                descriptors = torch.cat((k_pool.unsqueeze(3), extremes), dim=3).reshape(
                    1, 1, 4 * (tail_rank + 1), 128
                )
                logits = (
                    torch.baddbmm(
                        torch.zeros(
                            (1, 4, 4 * (tail_rank + 1)),
                            device="cuda",
                            dtype=torch.float16,
                        ),
                        q_pool.flatten(0, 1),
                        descriptors.flatten(0, 1).transpose(-1, -2),
                        beta=0,
                        alpha=1 / math.sqrt(128),
                    )
                    .view(1, 1, 4, 4, tail_rank + 1)
                    .float()
                )
                mean = logits[..., 0]
                tails = logits[..., 1:]
                bulk = (key_counts * mean - tails.sum(-1)) / (key_counts - tail_rank)
                reference = (
                    torch.logsumexp(
                        torch.cat(
                            (
                                (bulk + (key_counts - tail_rank).log()).unsqueeze(-1),
                                tails,
                            ),
                            dim=-1,
                        ),
                        dim=-1,
                    )
                    .softmax(-1)
                    .half()
                )

                actual = operation(q_pool, k_pool, packed_k, counts, prefix_blocks=0)
                torch.testing.assert_close(actual.float(), reference.float(), atol=2.5e-4, rtol=0)
                torch.testing.assert_close(
                    actual.float().sum(-1),
                    torch.ones_like(actual.float().sum(-1)),
                    atol=5e-4,
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
