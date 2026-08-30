from __future__ import annotations

import os
import unittest
from importlib.util import find_spec

import torch

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
    @torch.inference_mode()
    def test_empty_route_ctas_define_output_and_lse_for_q64_tail_and_q128(self) -> None:
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

                self.assertTrue(torch.isfinite(output[:, :, :query_block]).all())
                self.assertTrue(torch.isfinite(lse[:, :, :query_block]).all())
                self.assertTrue(
                    torch.equal(
                        output[:, :, query_block:],
                        torch.zeros_like(output[:, :, query_block:]),
                    )
                )
                self.assertTrue(torch.isneginf(lse[:, :, query_block:]).all())

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

                    self.assertTrue(torch.isfinite(output[:, :, :query_block]).all())
                    self.assertTrue(torch.isfinite(lse[:, :, :query_block]).all())
                    self.assertTrue(
                        torch.equal(
                            output[:, :, query_block:],
                            torch.zeros_like(output[:, :, query_block:]),
                        )
                    )
                    self.assertTrue(torch.isneginf(lse[:, :, query_block:]).all())

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
        reference: tuple[torch.Tensor, torch.Tensor] | None = None

        for iteration in range(iterations):
            output, lse = sm120_q64_fp16_attention(
                query, key, value, block_ids, block_counts, valid_k_counts
            )
            self.assertTrue(torch.isfinite(output[:, :, :64]).all())
            self.assertTrue(torch.isfinite(lse[:, :, :64]).all())
            self.assertTrue(torch.equal(output[:, :, 64:], torch.zeros_like(output[:, :, 64:])))
            self.assertTrue(torch.isneginf(lse[:, :, 64:]).all())
            actual = output.cpu(), lse.cpu()
            if reference is None:
                reference = actual
            else:
                self.assertTrue(
                    torch.equal(actual[0], reference[0]) and torch.equal(actual[1], reference[1]),
                    f"allocator-history-dependent output at iteration {iteration}",
                )
            del output, lse, actual
            churn_output = torch.full_like(query, torch.nan)
            churn_lse = torch.full(query.shape[:-1], torch.nan, device="cuda", dtype=torch.float32)
            del churn_output, churn_lse


if __name__ == "__main__":
    unittest.main()
