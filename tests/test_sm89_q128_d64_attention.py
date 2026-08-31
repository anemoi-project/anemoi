from __future__ import annotations

import math
import unittest
from importlib.util import find_spec
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
_CUDA = torch.cuda.is_available()
_CAPABILITY = torch.cuda.get_device_capability() if _CUDA else None
_EXTENSION = find_spec("anemoi.layers.attention.mpa._cuda_attention") is not None


class SM89Q128D64BuildTests(unittest.TestCase):
    def test_build_includes_each_q128_d64_phase(self) -> None:
        setup = (ROOT / "setup.py").read_text()
        for name in (
            "inst_q128_k64_d64.cu",
            "inst_q128_k64_d64_fp16.cu",
            "inst_q128_k64_d64_int8.cu",
        ):
            self.assertIn(f'sm89 / "instantiations" / "{name}"', setup)

    def test_q128_d64_instantiations_match_the_supported_phases(self) -> None:
        directory = ROOT / "csrc/attention/cuda/sm89/instantiations"
        expected = {
            "inst_q128_k64_d64.cu": "<64, true, true, false>",
            "inst_q128_k64_d64_fp16.cu": "<64, false, true, false>",
            "inst_q128_k64_d64_int8.cu": "<64, true, false, false>",
        }
        for name, signature in expected.items():
            source = (directory / name).read_text()
            self.assertIn("#define MPA_CTA_Q 128", source)
            self.assertIn("#define MPA_WARP_Q 32", source)
            self.assertIn(signature, source)

    def test_host_limits_d64_to_non_smooth_q128(self) -> None:
        source = (
            ROOT / "csrc/attention/cuda/sm89/k64_attention_host.cu"
        ).read_text()
        self.assertIn("QueryBlock == 128 && query.size(3) == 64", source)
        self.assertIn(
            "!SmoothK && QueryBlock == 128 && q16.size(3) == 64", source
        )
        self.assertIn(
            "launch_mixed_attention_sm89_q128_k64<64, true, true, false>",
            source,
        )


@unittest.skipUnless(
    _CUDA and _CAPABILITY == (8, 9) and _EXTENSION,
    "requires the native attention extension on SM89",
)
class SM89Q128D64CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from anemoi.layers.attention.mpa.backends.sm89_k64 import (
            native_k64_mixed_attention,
            prepare_k64_fp8_operands,
        )

        cls.native_attention = staticmethod(native_k64_mixed_attention)
        cls.prepare_operands = staticmethod(prepare_k64_fp8_operands)

    def setUp(self) -> None:
        torch.manual_seed(20260831)
        self.query = torch.randn(
            (1, 4, 256, 64), device="cuda", dtype=torch.float16
        )
        self.key = torch.randn(
            (1, 2, 192, 64), device="cuda", dtype=torch.float16
        )
        self.value = torch.randn_like(self.key)
        self.prepared = self.prepare_operands(
            self.query,
            self.key,
            self.value,
            query_block=128,
        )
        # Only five lanes in the final K64 block are valid. Poisoning its tail
        # makes a missing valid_k_counts mask fail loudly. Quantized operands
        # are prepared first so invalid lanes cannot distort their scales.
        self.key[:, :, 133:] = 64
        self.value[:, :, 133:] = 64
        self.route = torch.zeros(
            (1, 4, 2, 3), device="cuda", dtype=torch.int32
        )
        self.route[:, :, 0, 0] = 0
        self.route[:, :, 0, 1] = 2
        self.valid_k_counts = torch.tensor(
            [[64, 64, 5]], device="cuda", dtype=torch.int32
        )

    def _reference_first_query_block(self) -> torch.Tensor:
        indices = torch.cat(
            (
                torch.arange(0, 64, device="cuda"),
                torch.arange(128, 133, device="cuda"),
            )
        )
        key = self.key.index_select(2, indices).repeat_interleave(2, dim=1)
        value = self.value.index_select(2, indices).repeat_interleave(2, dim=1)
        score = torch.matmul(
            self.query[:, :, :128].float(), key.float().transpose(-1, -2)
        ) / math.sqrt(64)
        return torch.matmul(score.softmax(dim=-1), value.float()).half()

    def _run_phase(
        self,
        low_blocks: int,
        high_blocks: int,
        *,
        active_int8: bool,
        active_fp16: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low_counts = torch.zeros(
            (1, 4, 2), device="cuda", dtype=torch.int32
        )
        high_counts = torch.zeros_like(low_counts)
        low_counts[:, :, 0] = low_blocks
        high_counts[:, :, 0] = high_blocks
        return self.native_attention(
            self.query,
            self.key,
            self.value,
            self.route,
            low_counts,
            self.route,
            high_counts,
            self.valid_k_counts,
            prepared_operands=self.prepared,
            active_int8=active_int8,
            active_fp16=active_fp16,
            query_block=128,
        )

    def _assert_phase(
        self,
        output: torch.Tensor,
        lse: torch.Tensor,
        relative_l2_limit: float,
    ) -> None:
        reference = self._reference_first_query_block()
        actual = output[:, :, :128]
        relative_l2 = (
            (actual.float() - reference.float()).square().sum().sqrt()
            / reference.float().square().sum().sqrt()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), reference.float().flatten(), dim=0
        ).item()
        self.assertLess(relative_l2, relative_l2_limit)
        self.assertGreater(cosine, 0.99)
        self.assertTrue(torch.isfinite(lse[:, :, :128]).all())
        self.assertTrue(
            torch.equal(output[:, :, 128:], torch.zeros_like(output[:, :, 128:]))
        )
        self.assertTrue(torch.isneginf(lse[:, :, 128:]).all())

    @torch.inference_mode()
    def test_fp16_phase_matches_reference_with_gqa_tail_and_empty_row(self) -> None:
        output, lse = self._run_phase(
            0, 2, active_int8=False, active_fp16=True
        )
        self._assert_phase(output, lse, 0.003)

    @torch.inference_mode()
    def test_int8_phase_is_bounded_with_gqa_tail_and_empty_row(self) -> None:
        output, lse = self._run_phase(
            2, 0, active_int8=True, active_fp16=False
        )
        self._assert_phase(output, lse, 0.10)

    @torch.inference_mode()
    def test_mixed_phase_is_bounded_with_gqa_tail_and_empty_row(self) -> None:
        output, lse = self._run_phase(
            1, 1, active_int8=True, active_fp16=True
        )
        self._assert_phase(output, lse, 0.10)


if __name__ == "__main__":
    unittest.main()
