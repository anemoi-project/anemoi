from __future__ import annotations

import os
import unittest
from importlib.util import find_spec

import torch

from anemoi.layers.attention.mpa.backends.sm89_k64 import assemble_h3_k64_output
from anemoi.layers.attention.mpa.backends.sm120_q64 import (
    prepare_h3_sm120_operands,
    sm120_q64_fp16_attention,
)
from anemoi.layers.attention.mpa.backends.sm120_q128 import (
    sm120_q128_fp16_attention,
)
from anemoi.layers.attention.mpa.executor import _run_sm120_phases
from anemoi.models.minimax_h3.native_k64_attention import (
    materialize_ragged_2d_layout,
)

_CUDA = torch.cuda.is_available()
_CAPABILITY = torch.cuda.get_device_capability() if _CUDA else None
_EXTENSION = find_spec("anemoi.layers.attention.mpa._cuda_sm120_q64") is not None


@unittest.skipUnless(
    _CUDA and _CAPABILITY == (12, 0) and _EXTENSION,
    "requires the native SM120 attention extension",
)
class SM120LifecycleCudaTests(unittest.TestCase):
    @staticmethod
    def _assemble_video(
        video: torch.Tensor,
        query_block: int,
        low_counts: torch.Tensor,
        middle_counts: torch.Tensor,
        high_counts: torch.Tensor,
    ) -> torch.Tensor:
        video_tokens = video.size(2)
        video_capacity = low_counts.size(2) * query_block
        if video_tokens != video_capacity:
            padded = torch.full(
                (*video.shape[:2], video_capacity, video.shape[3]),
                torch.nan,
                device=video.device,
                dtype=video.dtype,
            )
            padded[:, :, :video_tokens].copy_(video)
            video = padded
        output = assemble_h3_k64_output(
            torch.zeros(
                (*video.shape[:2], 1, video.shape[3]),
                device=video.device,
                dtype=video.dtype,
            ),
            video,
            torch.arange(video_tokens, device=video.device, dtype=torch.int64),
            route_counts=(low_counts, middle_counts, high_counts),
            query_block_size=query_block,
        )
        return output[:, 1:]

    @torch.inference_mode()
    def test_empty_aware_assembly_does_not_read_poisoned_video_rows(self) -> None:
        for query_block in (64, 128):
            for output_dtype in (torch.float16, torch.bfloat16):
                with self.subTest(query_block=query_block, output_dtype=output_dtype):
                    video = torch.full(
                        (1, 1, 2 * query_block, 128),
                        torch.nan,
                        device="cuda",
                        dtype=torch.float16,
                    )
                    video[:, :, :query_block].fill_(1.0)
                    counts = torch.zeros(
                        (1, 1, 2), device="cuda", dtype=torch.int32
                    )
                    counts[:, :, 0] = 1
                    output = assemble_h3_k64_output(
                        torch.ones(
                            (1, 1, 1, 128), device="cuda", dtype=output_dtype
                        ),
                        video,
                        torch.arange(
                            2 * query_block, device="cuda", dtype=torch.int64
                        ),
                        output_dtype=output_dtype,
                        route_counts=(counts, torch.zeros_like(counts), torch.zeros_like(counts)),
                        query_block_size=query_block,
                    )
                    nonempty = output[:, 1 : 1 + query_block]
                    empty = output[:, 1 + query_block :]
                    self.assertTrue(torch.equal(nonempty, torch.ones_like(nonempty)))
                    self.assertTrue(torch.equal(empty, torch.zeros_like(empty)))
                    self.assertFalse(torch.signbit(empty).any())

    @torch.inference_mode()
    def test_empty_route_ctas_define_output_and_return_empty_lse(self) -> None:
        cases = (
            (64, 96, sm120_q64_fp16_attention),
            (128, 256, sm120_q128_fp16_attention),
        )
        for query_block, query_tokens, attention in cases:
            with self.subTest(query_block=query_block):
                query = (
                    torch.linspace(
                        -1,
                        1,
                        steps=query_tokens * 128,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    .reshape(1, 1, query_tokens, 128)
                    .half()
                )
                key = (
                    torch.linspace(
                        1,
                        -1,
                        steps=128 * 128,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    .reshape(1, 1, 128, 128)
                    .half()
                )
                value = key.flip(-1).contiguous()
                block_ids = (
                    torch.arange(2, device="cuda", dtype=torch.int32)
                    .view(1, 1, 1, 2)
                    .expand(1, 1, 2, 2)
                    .contiguous()
                )
                block_counts = torch.tensor([[[2, 0]]], device="cuda", dtype=torch.int32)
                valid_k_counts = torch.full((1, 2), 64, device="cuda", dtype=torch.int32)

                poison_output = torch.full_like(query, torch.nan)
                poison_lse = torch.full(
                    query.shape[:-1], torch.nan, device="cuda", dtype=torch.float32
                )
                del poison_output, poison_lse

                output, lse = attention(
                    query,
                    key,
                    value,
                    block_ids,
                    block_counts,
                    valid_k_counts,
                )
                zero_counts = torch.zeros_like(block_counts)
                final = self._assemble_video(
                    output,
                    query_block,
                    zero_counts,
                    zero_counts,
                    block_counts,
                )

                self.assertTrue(torch.isfinite(final[:, :query_block]).all())
                self.assertTrue(lse.is_cuda)
                self.assertEqual(lse.dtype, torch.float32)
                self.assertEqual(lse.numel(), 0)
                self.assertTrue(
                    torch.equal(
                        final[:, query_block:],
                        torch.zeros_like(final[:, query_block:]),
                    )
                )
                self.assertFalse(torch.signbit(final[:, query_block:]).any())

    @torch.inference_mode()
    def test_empty_route_ctas_cover_every_compiled_precision_family(self) -> None:
        ratios_matrix = (
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
        phase_names = ("nvfp4", "int8", "mxfp8", "fp16")
        raw = (
            torch.linspace(
                -1,
                1,
                steps=256 * 128,
                device="cuda",
                dtype=torch.float32,
            )
            .reshape(1, 1, 256, 128)
            .half()
        )
        value = raw.flip(-1).contiguous()
        scales = tuple(torch.ones((), device="cuda", dtype=torch.float32) for _ in range(3))

        for query_block in (64, 128):
            layout = materialize_ragged_2d_layout(
                torch.device("cuda"),
                frames=1,
                height=16,
                width=16,
                logical_block=query_block,
                enable_anchors=False,
            )
            prepared = prepare_h3_sm120_operands(
                raw,
                raw,
                value,
                layout.indices,
                layout.slot_valid,
                layout.counts,
                prefix_tokens=0,
                query_block_size=query_block,
                has_nvfp4=True,
                has_int8=True,
                has_mxfp8=True,
                has_fp16=True,
                has_prefix_query_int8=False,
                has_maxpool=False,
                global_scales=scales,
            )
            query_fp16, key_fp16, value_fp16 = prepared[2:5]
            query_blocks = query_fp16.size(2) // query_block
            block_ids = (
                torch.arange(4, device="cuda", dtype=torch.int32)
                .view(1, 1, 1, 4)
                .expand(1, 1, query_blocks, 4)
                .contiguous()
            )
            valid_k_counts = torch.full((1, 4), 64, device="cuda", dtype=torch.int32)

            for ratios in ratios_matrix:
                with self.subTest(query_block=query_block, ratios=ratios):
                    active = [
                        phase for phase, ratio in zip(phase_names, ratios, strict=True) if ratio
                    ]
                    stage_counts = (
                        (4,) if len(active) == 1 else (2, 2) if len(active) == 2 else (1, 1, 2)
                    )
                    counts = {
                        phase: torch.zeros((1, 1, query_blocks), device="cuda", dtype=torch.int32)
                        for phase in phase_names
                    }
                    for phase, count in zip(active, stage_counts, strict=True):
                        counts[phase][0, 0, 0] = count
                    fp16_counts = (
                        counts["fp16"]
                        if ratios[3]
                        else torch.empty(0, device="cuda", dtype=torch.int32)
                    )

                    poison_output = torch.full_like(query_fp16, torch.nan)
                    poison_lse = torch.full(
                        query_fp16.shape[:-1],
                        torch.nan,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    del poison_output, poison_lse

                    output, lse = _run_sm120_phases(
                        query_block_size=query_block,
                        ratios=ratios,
                        query_fp16=query_fp16,
                        key_fp16=key_fp16,
                        value_fp16=value_fp16,
                        block_ids=block_ids,
                        nvfp4_counts=counts["nvfp4"],
                        middle_counts=counts["int8"] + counts["mxfp8"],
                        fp16_counts=fp16_counts,
                        valid_k_counts=valid_k_counts,
                        layer=0,
                        fp16_prefix_blocks=0,
                        prepared_nv_operands=prepared[5:11],
                        prepared_mxfp8_operands=prepared[11:17],
                        prepared_int8_operands=prepared[17:23],
                        prepared_global_scales=scales,
                    )
                    final = self._assemble_video(
                        output,
                        query_block,
                        counts["nvfp4"],
                        counts["int8"] + counts["mxfp8"],
                        fp16_counts,
                    )

                    self.assertTrue(torch.isfinite(final[:, :query_block]).all())
                    self.assertTrue(lse.is_cuda)
                    self.assertEqual(lse.dtype, torch.float32)
                    self.assertEqual(lse.numel(), 0)
                    self.assertTrue(
                        torch.equal(
                            final[:, query_block:],
                            torch.zeros_like(final[:, query_block:]),
                        )
                    )
                    self.assertFalse(torch.signbit(final[:, query_block:]).any())

    @torch.inference_mode()
    def test_empty_route_output_is_stable_in_a_long_lived_context(self) -> None:
        iterations = int(os.environ.get("ANEMOI_SM120_LIFECYCLE_ITERS", "256"))
        self.assertGreaterEqual(iterations, 256)
        query = torch.ones((1, 1, 96, 128), device="cuda", dtype=torch.float16)
        key = torch.ones((1, 1, 128, 128), device="cuda", dtype=torch.float16)
        value = (
            torch.linspace(
                -1,
                1,
                steps=128 * 128,
                device="cuda",
                dtype=torch.float32,
            )
            .reshape_as(key)
            .half()
        )
        block_ids = (
            torch.arange(2, device="cuda", dtype=torch.int32)
            .view(1, 1, 1, 2)
            .expand(1, 1, 2, 2)
            .contiguous()
        )
        block_counts = torch.tensor([[[2, 0]]], device="cuda", dtype=torch.int32)
        valid_k_counts = torch.full((1, 2), 64, device="cuda", dtype=torch.int32)
        reference: torch.Tensor | None = None

        for iteration in range(iterations):
            output, lse = sm120_q64_fp16_attention(
                query, key, value, block_ids, block_counts, valid_k_counts
            )
            zero_counts = torch.zeros_like(block_counts)
            final = self._assemble_video(
                output, 64, zero_counts, zero_counts, block_counts
            )
            self.assertTrue(torch.isfinite(final[:, :64]).all())
            self.assertTrue(lse.is_cuda)
            self.assertEqual(lse.dtype, torch.float32)
            self.assertEqual(lse.numel(), 0)
            self.assertTrue(torch.equal(final[:, 64:], torch.zeros_like(final[:, 64:])))
            self.assertFalse(torch.signbit(final[:, 64:]).any())
            actual = final.cpu()
            if reference is None:
                reference = actual
            else:
                self.assertTrue(
                    torch.equal(actual, reference),
                    f"allocator-history-dependent output at iteration {iteration}",
                )
            del output, final, lse, actual
            churn_output = torch.full_like(query, torch.nan)
            churn_lse = torch.full(query.shape[:-1], torch.nan, device="cuda", dtype=torch.float32)
            del churn_output, churn_lse


if __name__ == "__main__":
    unittest.main()
