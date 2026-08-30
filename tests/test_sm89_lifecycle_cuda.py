from __future__ import annotations

import os
import unittest
from importlib.util import find_spec

import torch

import anemoi
from anemoi.layers.attention.mpa.backends.sm89_k64 import (
    native_k64_mixed_attention,
    prepare_k64_fp8_operands,
)


_CUDA = torch.cuda.is_available()
_CAPABILITY = torch.cuda.get_device_capability() if _CUDA else None
_EXTENSION = find_spec("anemoi.layers.attention.mpa._cuda_attention") is not None


@unittest.skipUnless(
    _CUDA and _CAPABILITY == (8, 9) and _EXTENSION,
    "requires the native attention extension on SM89",
)
class SM89LifecycleCudaTests(unittest.TestCase):
    @torch.inference_mode()
    def test_empty_route_ctas_have_defined_output_for_every_phase_family(self) -> None:
        """An empty global-top-k row must not expose torch::empty contents."""

        torch.manual_seed(41)
        for query_block in (64, 128):
            query = torch.randn(
                (1, 2, 2 * query_block, 128),
                device="cuda",
                dtype=torch.float16,
            )
            key = torch.randn((1, 2, 128, 128), device="cuda", dtype=torch.float16)
            value = torch.randn_like(key)
            prepared = prepare_k64_fp8_operands(
                query,
                key,
                value,
                query_block=query_block,
            )
            route = (
                torch.arange(2, device="cuda", dtype=torch.int32)
                .view(1, 1, 1, 2)
                .expand(1, 2, 2, 2)
                .contiguous()
            )
            valid = torch.full((1, 2), 64, device="cuda", dtype=torch.int32)
            empty = torch.zeros((1, 2, 2), device="cuda", dtype=torch.int32)

            cases = (
                ("int8", (2, 0), True, False),
                ("fp16", (0, 2), False, True),
                ("mixed", (1, 1), True, True),
            )
            for phase, (low_count, high_count), active_int8, active_fp16 in cases:
                with self.subTest(query_block=query_block, phase=phase):
                    low = empty.clone()
                    high = empty.clone()
                    low[:, :, 0] = low_count
                    high[:, :, 0] = high_count

                    # Make the old failure deterministic: the native host wrapper
                    # allocates an output with the same shape immediately below.
                    poison = torch.full_like(query, torch.nan)
                    del poison
                    output, lse = native_k64_mixed_attention(
                        query,
                        key,
                        value,
                        route,
                        low,
                        route,
                        high,
                        valid,
                        prepared_operands=prepared,
                        active_int8=active_int8,
                        active_fp16=active_fp16,
                        query_block=query_block,
                    )

                    self.assertTrue(torch.isfinite(output[:, :, :query_block]).all())
                    empty_output = output[:, :, query_block:]
                    self.assertTrue(
                        torch.equal(empty_output, torch.zeros_like(empty_output))
                    )
                    self.assertTrue(torch.isfinite(lse[:, :, :query_block]).all())
                    self.assertTrue(torch.isneginf(lse[:, :, query_block:]).all())

    @torch.inference_mode()
    def test_public_api_survives_long_lived_context_with_empty_global_rows(self) -> None:
        """Exercise allocator reuse in one context for longer than the LTX failure window."""

        iterations = int(os.environ.get("ANEMOI_SM89_LIFECYCLE_ITERS", "256"))
        self.assertGreaterEqual(iterations, 256)
        shape = (1, 256, 1, 128)
        query = torch.ones(shape, device="cuda", dtype=torch.bfloat16)
        key = torch.ones_like(query)
        torch.manual_seed(43)
        value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        sparse = anemoi.SparseConfig(query_block_size=64, sparsity_ratio=0.80)
        quant = anemoi.QuantConfig(int8_ratio=1.0)
        layout = anemoi.VisualLayout((1, 16, 16))
        reference: torch.Tensor | None = None

        for iteration in range(iterations):
            output = anemoi.anemoi_attention(
                query,
                key,
                value,
                layout=layout,
                layer=iteration,
                sparse_config=sparse,
                quant_config=quant,
            )
            self.assertTrue(
                torch.isfinite(output).all(),
                f"non-finite SM89 output at lifecycle iteration {iteration}",
            )
            # Uniform DraftMap probabilities and stable global top-k spend the
            # three retained cells in the first of four rows.  The other three
            # rows therefore exercise the empty-route CTA contract.
            self.assertGreaterEqual(
                int(output.eq(0).all(dim=-1).sum()),
                3 * 64,
            )
            output_cpu = output.cpu()
            if reference is None:
                reference = output_cpu
            else:
                self.assertTrue(
                    torch.equal(output_cpu, reference),
                    f"allocator-history-dependent output at iteration {iteration}",
                )
            del output, output_cpu

            # Reuse and poison same-sized caching-allocator blocks between API
            # calls.  The kernel must overwrite every logically defined row.
            churn = torch.full(shape, torch.nan, device="cuda", dtype=torch.bfloat16)
            del churn


if __name__ == "__main__":
    unittest.main()
