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


class SM89D64BuildTests(unittest.TestCase):
    def test_build_includes_every_d64_kernel_family(self) -> None:
        setup = (ROOT / "setup.py").read_text()
        for name in (
            "inst_k64_d64.cu",
            "inst_k64_d64_int8_dense.cu",
            "inst_q128_k64_d64.cu",
            "inst_q128_k64_d64_fp16.cu",
            "inst_q128_k64_d64_int8.cu",
        ):
            self.assertIn(f'sm89 / "instantiations" / "{name}"', setup)

    def test_q64_d64_instantiates_every_public_sparse_phase(self) -> None:
        source = (
            ROOT
            / "csrc/attention/cuda/sm89/instantiations/inst_k64_d64.cu"
        ).read_text()
        for signature in (
            "<64, false, true, false>",
            "<64, true, true, false>",
            "<64, true, true, true>",
            "<64, true, false, false>",
        ):
            self.assertIn(signature, source)

    def test_q128_d64_instantiates_every_public_sparse_phase(self) -> None:
        directory = ROOT / "csrc/attention/cuda/sm89/instantiations"
        expected = {
            "inst_q128_k64_d64.cu": (
                "<64, true, true, false>",
                "<64, true, true, true>",
            ),
            "inst_q128_k64_d64_fp16.cu": ("<64, false, true, false>",),
            "inst_q128_k64_d64_int8.cu": ("<64, true, false, false>",),
        }
        for name, signatures in expected.items():
            source = (directory / name).read_text()
            self.assertIn("#define MPA_CTA_Q 128", source)
            self.assertIn("#define MPA_WARP_Q 32", source)
            for signature in signatures:
                self.assertIn(signature, source)

    def test_all_sparse_d64_launches_inherit_no_lse_contract(self) -> None:
        source = (
            ROOT / "csrc/attention/cuda/sm89/k64_attention_host.cu"
        ).read_text()
        self.assertIn("query.size(3) == 64 || query.size(3) == 128", source)
        self.assertIn("q16.size(3) == 64 || q16.size(3) == 128", source)
        self.assertIn("q8.size(3) == 64 || q8.size(3) == 128", source)
        self.assertNotIn("lse.data_ptr<float>()", source)

    def test_prefix_quantization_and_scale_are_head_dim_generic(self) -> None:
        producer = (
            ROOT / "csrc/attention/cuda/sm89/raster_preprocess.cu"
        ).read_text()
        backend = (
            ROOT / "anemoi/layers/attention/mpa/backends/sm89_k64.py"
        ).read_text()
        self.assertIn("prepare_sm89_prefix_q_int8_kernel<InputT, HeadDim>", producer)
        self.assertIn("tensor.size(3) == 64 || tensor.size(3) == 128", producer)
        self.assertIn('"prepare_sm89_prefix_q_int8"', backend)
        self.assertIn("1.0 / math.sqrt(prefix_q8.size(-1))", backend)


@unittest.skipUnless(
    _CUDA and _CAPABILITY == (8, 9) and _EXTENSION,
    "requires the native attention extension on SM89",
)
class SM89D64CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from anemoi.layers.attention.mpa.backends.sm89_k64 import (
            assemble_h3_k64_output,
            native_k64_mixed_attention,
            prepare_k64_fp8_operands,
            prepare_prefix_q_int8,
            quantize_qk_k64,
            sm89_q64_prefix_int8_attention,
        )

        cls.assemble = staticmethod(assemble_h3_k64_output)
        cls.native_attention = staticmethod(native_k64_mixed_attention)
        cls.prepare_operands = staticmethod(prepare_k64_fp8_operands)
        cls.prepare_prefix = staticmethod(prepare_prefix_q_int8)
        cls.prefix_attention = staticmethod(sm89_q64_prefix_int8_attention)
        cls.quantize_qk = staticmethod(quantize_qk_k64)

    @torch.inference_mode()
    def test_native_contiguous_qk_quantizer_preserves_contract_and_tail(self) -> None:
        def reference(
            tensor: torch.Tensor,
            block: int,
            mean: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if mean is not None:
                tensor = (tensor.float() - mean.unsqueeze(2).float()).half()
            batch, heads, tokens, head_dim = tensor.shape
            blocks = (tokens + block - 1) // block
            padded = torch.zeros(
                (batch, heads, blocks * block, head_dim),
                device="cuda",
                dtype=torch.float16,
            )
            padded[:, :, :tokens] = tensor
            values = padded.view(batch, heads, blocks, block, head_dim).float()
            scales = values.abs().amax(dim=(-1, -2)) / 127.0 + 1.0e-7
            scaled = values / scales[..., None, None]
            quantized = (
                scaled + 0.5 * torch.where(scaled >= 0, 1, -1)
            ).to(torch.int8)
            return quantized.view_as(padded)[:, :, :tokens], scales

        torch.manual_seed(20260903)
        for head_dim in (64, 128):
            for query_block in (64, 128):
                query = torch.randn(
                    (2, 2, query_block + 7, head_dim),
                    device="cuda",
                    dtype=torch.float16,
                )
                key = torch.randn(
                    (2, 1, 73, head_dim), device="cuda", dtype=torch.float16
                )
                key_mean = key.mean(dim=2).contiguous()
                for mean in (None, key_mean):
                    with self.subTest(
                        head_dim=head_dim,
                        query_block=query_block,
                        smooth=mean is not None,
                    ):
                        q8, q_scale, k8, k_scale = self.quantize_qk(
                            query, key, mean, query_block, 64
                        )
                        expected_q8, expected_q_scale = reference(
                            query, query_block
                        )
                        expected_k8, expected_k_scale = reference(key, 64, mean)
                        self.assertTrue(torch.equal(q8, expected_q8))
                        self.assertTrue(torch.equal(k8, expected_k8))
                        torch.testing.assert_close(
                            q_scale, expected_q_scale, rtol=1.0e-6, atol=1.0e-9
                        )
                        torch.testing.assert_close(
                            k_scale, expected_k_scale, rtol=1.0e-6, atol=1.0e-9
                        )

    def _make_case(self, query_block: int) -> dict[str, object]:
        torch.manual_seed(20260902 + query_block)
        query = torch.randn(
            (2, 4, 2 * query_block, 64), device="cuda", dtype=torch.float16
        )
        key = torch.randn(
            (2, 2, 192, 64), device="cuda", dtype=torch.float16
        )
        value = torch.randn_like(key)
        regular = self.prepare_operands(
            query, key, value, query_block=query_block
        )
        key_mean = key.mean(dim=2).contiguous()
        q8, q_scale, k8, k_scale = self.quantize_qk(
            query, key, key_mean, query_block, 64
        )
        smooth = (
            q8,
            k8,
            regular[2],
            q_scale,
            k_scale,
            regular[5],
            key_mean,
        )

        # Quantize first, then poison invalid lanes so the native FP16 mask and
        # pre-quantized INT8 path are both independent of tail scale drift.
        key[:, :, 133:] = 64
        value[:, :, 133:] = 64
        valid = torch.tensor(
            [[64, 64, 5], [64, 64, 5]], device="cuda", dtype=torch.int32
        )
        return {
            "query": query,
            "key": key,
            "value": value,
            "regular": regular,
            "smooth": smooth,
            "valid": valid,
        }

    def _run_phase(
        self, case: dict[str, object], query_block: int, phase: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = case["query"]
        key = case["key"]
        value = case["value"]
        valid = case["valid"]
        assert isinstance(query, torch.Tensor)
        assert isinstance(key, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        assert isinstance(valid, torch.Tensor)

        route = (
            torch.arange(3, device="cuda", dtype=torch.int32)
            .view(1, 1, 1, 3)
            .expand(2, 4, 2, 3)
            .contiguous()
        )
        low = torch.zeros((2, 4, 2), device="cuda", dtype=torch.int32)
        high = torch.zeros_like(low)
        prefix_blocks = 0
        active_int8 = phase != "fp16"
        active_fp16 = phase in ("fp16", "mixed", "smooth_mixed")
        prepared = case["smooth"] if phase.startswith("smooth") else case["regular"]

        if phase == "fp16":
            high[:, :, 0] = 3
        elif phase in ("int8", "smooth_int8"):
            low[:, :, 0] = 3
        else:
            # Exact block 0 is implicit; the aliased route stores low block 1
            # followed by FP16 rescue block 2.
            route[:, :, 0, 0] = 1
            route[:, :, 0, 1] = 2
            low[:, :, 0] = 1
            # FP16 counts include the implicit prefix stage even though its ID
            # is synthesized by the kernel rather than stored in the route.
            high[:, :, 0] = 2
            prefix_blocks = 1

        output, lse = self.native_attention(
            query,
            key,
            value,
            route,
            low,
            route,
            high,
            valid,
            fp16_prefix_blocks=prefix_blocks,
            prepared_operands=prepared,
            active_int8=active_int8,
            active_fp16=active_fp16,
            query_block=query_block,
        )
        return output, lse, low, high

    def _reference(self, case: dict[str, object], query_block: int) -> torch.Tensor:
        query = case["query"]
        key = case["key"]
        value = case["value"]
        assert isinstance(query, torch.Tensor)
        assert isinstance(key, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        key = key[:, :, :133].repeat_interleave(2, dim=1)
        value = value[:, :, :133].repeat_interleave(2, dim=1)
        scores = torch.matmul(
            query[:, :, :query_block].float(), key.float().transpose(-1, -2)
        ) / math.sqrt(64)
        return torch.matmul(scores.softmax(dim=-1), value.float()).half()

    def _assert_phase(
        self,
        case: dict[str, object],
        query_block: int,
        phase: str,
        relative_l2_limit: float,
    ) -> None:
        output, lse, low, high = self._run_phase(case, query_block, phase)
        reference = self._reference(case, query_block)
        actual = output[:, :, :query_block]
        relative_l2 = (
            (actual.float() - reference.float()).square().sum().sqrt()
            / reference.float().square().sum().sqrt()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), reference.float().flatten(), dim=0
        ).item()
        self.assertLess(relative_l2, relative_l2_limit)
        self.assertGreater(cosine, 0.99)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(lse.is_cuda)
        self.assertEqual(lse.dtype, torch.float32)
        self.assertEqual(lse.numel(), 0)

        # Native empty rows are intentionally undefined after the Pareto
        # writeback. The route-aware assembly boundary owns positive-zero output.
        output[:, :, query_block:].fill_(torch.nan)
        query = case["query"]
        assert isinstance(query, torch.Tensor)
        assembled = self.assemble(
            query[:, :, :0],
            output,
            torch.arange(
                query_block + 17, device="cuda", dtype=torch.int64
            ),
            route_counts=(low, high, None),
            query_block_size=query_block,
        )
        torch.testing.assert_close(
            assembled[:, :query_block].permute(0, 2, 1, 3),
            actual,
            rtol=0,
            atol=0,
        )
        empty_output = assembled[:, query_block:]
        self.assertTrue(torch.equal(empty_output, torch.zeros_like(empty_output)))
        self.assertFalse(torch.signbit(empty_output).any())

    @torch.inference_mode()
    def test_q64_q128_all_public_sparse_phases(self) -> None:
        limits = {
            "fp16": 0.003,
            "int8": 0.10,
            "smooth_int8": 0.10,
            "mixed": 0.10,
            "smooth_mixed": 0.10,
        }
        for query_block in (64, 128):
            case = self._make_case(query_block)
            for phase, limit in limits.items():
                with self.subTest(query_block=query_block, phase=phase):
                    self._assert_phase(case, query_block, phase, limit)

    @torch.inference_mode()
    def test_q64_prefix_int8_supports_d64(self) -> None:
        torch.manual_seed(20260966)
        prefix_tokens = 45
        query = torch.randn(
            (1, 4, prefix_tokens, 64), device="cuda", dtype=torch.float16
        )
        key = torch.randn((1, 2, 128, 64), device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
        padded_query = torch.empty(
            (1, 4, 64, 64), device="cuda", dtype=torch.float16
        )
        shared = self.prepare_operands(padded_query, key, value, query_block=64)
        prefix_q8, prefix_scale = self.prepare_prefix(query, prefix_tokens)
        valid = torch.full((1, 2), 64, device="cuda", dtype=torch.int32)
        output = self.prefix_attention(
            prefix_q8,
            prefix_scale,
            shared,
            valid,
            prefix_tokens,
        )

        expanded_key = key.repeat_interleave(2, dim=1)
        expanded_value = value.repeat_interleave(2, dim=1)
        scores = torch.matmul(
            query.float(), expanded_key.float().transpose(-1, -2)
        ) / math.sqrt(64)
        reference = torch.matmul(
            scores.softmax(dim=-1), expanded_value.float()
        ).half()
        relative_l2 = (
            (output.float() - reference.float()).square().sum().sqrt()
            / reference.float().square().sum().sqrt()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        ).item()
        self.assertEqual(output.shape, query.shape)
        self.assertLess(relative_l2, 0.10)
        self.assertGreater(cosine, 0.99)

    @torch.inference_mode()
    def test_stable_public_entry_runs_d64_without_triton_preparation(self) -> None:
        import anemoi

        layout = anemoi.VisualLayout((1, 8, 16), prefix_tokens=64)
        for dtype in (torch.float16, torch.bfloat16):
            for query_block in (64, 128):
                with self.subTest(dtype=dtype, query_block=query_block):
                    torch.manual_seed(20261000 + query_block)
                    shape = (1, 192, 2, 64)
                    query = (torch.randn(shape, device="cuda") * 0.2).to(dtype)
                    key = (torch.randn(shape, device="cuda") * 0.2).to(dtype)
                    value = (torch.randn(shape, device="cuda") * 0.2).to(dtype)
                    output = anemoi.anemoi_attention(
                        query,
                        key,
                        value,
                        layout=layout,
                        layer=0,
                        sparse_config=anemoi.SparseConfig(
                            query_block_size=query_block,
                            sparsity_ratio=0.0,
                            maxpool_weight=0.5,
                        ),
                        scale=1.0 / math.sqrt(64),
                    )
                    reference = torch.nn.functional.scaled_dot_product_attention(
                        query.transpose(1, 2),
                        key.transpose(1, 2),
                        value.transpose(1, 2),
                    ).transpose(1, 2)
                    relative_l2 = (
                        torch.linalg.vector_norm(output.float() - reference.float())
                        / torch.linalg.vector_norm(reference.float())
                    ).item()
                    self.assertEqual(output.shape, query.shape)
                    self.assertEqual(output.dtype, query.dtype)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertLess(relative_l2, 0.06)


if __name__ == "__main__":
    unittest.main()
