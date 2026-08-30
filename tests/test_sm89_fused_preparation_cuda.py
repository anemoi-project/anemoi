from __future__ import annotations

import math
from importlib.util import find_spec
import unittest

import torch

from anemoi.layers.attention.mpa.backends.sm89_k64 import (
    native_k64_mixed_attention,
    prepare_h3_sm89_int8_operands,
    sm89_h3_draft_probability,
)


_CUDA = torch.cuda.is_available()
_CAPABILITY = torch.cuda.get_device_capability() if _CUDA else None
_EXTENSION = find_spec("anemoi.layers.attention.mpa._cuda_attention") is not None


def _layout(query_block: int):
    counts = [64, 53, 17] if query_block == 64 else [117, 17]
    indices: list[int] = []
    valid: list[bool] = []
    cursor = 0
    for count in counts:
        indices.extend(range(cursor, cursor + count))
        indices.extend([0] * (query_block - count))
        valid.extend([True] * count)
        valid.extend([False] * (query_block - count))
        cursor += count
    return (
        torch.tensor(indices, device="cuda", dtype=torch.int64),
        torch.tensor(valid, device="cuda", dtype=torch.bool),
        torch.tensor(counts, device="cuda", dtype=torch.int32),
    )


def _quantize_reference(tensor: torch.Tensor, block: int):
    shaped = tensor.view(*tensor.shape[:2], -1, block, tensor.size(-1))
    scale = shaped.float().abs().amax(dim=(3, 4)) / 127.0 + 1.0e-7
    expanded = scale.repeat_interleave(block, dim=2).unsqueeze(-1)
    normalized = tensor.float() / expanded
    quantized = (
        normalized + torch.where(normalized >= 0.0, 0.5, -0.5)
    ).to(torch.int8)
    return quantized, scale


@unittest.skipUnless(
    _CUDA and _CAPABILITY == (8, 9) and _EXTENSION,
    "requires the native attention extension on SM89",
)
class SM89FusedPreparationCudaTests(unittest.TestCase):
    @torch.inference_mode()
    def test_single_load_q64_q128_matches_unfused_numerical_contract(self) -> None:
        torch.manual_seed(17)
        prefix_tokens = 70
        video_tokens = 134
        for dtype in (torch.float16, torch.bfloat16):
            for query_block in (64, 128):
                with self.subTest(dtype=dtype, query_block=query_block):
                    indices, valid, counts = _layout(query_block)
                    shape = (1, 2, prefix_tokens + video_tokens, 128)
                    query = torch.randn(shape, device="cuda").to(dtype)
                    key = torch.randn(shape, device="cuda").to(dtype)
                    value = torch.randn(shape, device="cuda").to(dtype)
                    prepared = prepare_h3_sm89_int8_operands(
                        query,
                        key,
                        value,
                        indices,
                        valid,
                        counts,
                        prefix_tokens=prefix_tokens,
                        query_block_size=query_block,
                        smooth_k=False,
                    )
                    self.assertEqual(prepared[12].numel(), 0)
                    self.assertEqual(prepared[13].numel(), 0)
                    q_pool, k_pool, packed_q, packed_k = prepared[:4]
                    q8, k8, q_scale, k_scale = (
                        prepared[5],
                        prepared[6],
                        prepared[8],
                        prepared[9],
                    )

                    gathered_q = query[:, :, prefix_tokens + indices].to(torch.float16)
                    gathered_k = key[:, :, prefix_tokens + indices].to(torch.float16)
                    mask = valid.view(1, 1, -1, 1)
                    gathered_q = torch.where(mask, gathered_q, 0.0).contiguous()
                    gathered_k = torch.where(mask, gathered_k, 0.0).contiguous()
                    prefix_capacity = math.ceil(prefix_tokens / 64) * 64
                    prefix_k = torch.zeros(
                        (1, 2, prefix_capacity, 128),
                        device="cuda",
                        dtype=torch.float16,
                    )
                    prefix_k[:, :, :prefix_tokens] = key[:, :, :prefix_tokens]
                    expected_k = torch.cat((prefix_k, gathered_k), dim=2)
                    torch.testing.assert_close(packed_q, gathered_q, atol=0, rtol=0)
                    torch.testing.assert_close(packed_k, expected_k, atol=0, rtol=0)

                    denominator = counts.view(1, 1, -1, 1)
                    expected_q_pool = (
                        gathered_q.view(1, 2, -1, query_block, 128)
                        .float()
                        .sum(3)
                        .div(denominator)
                        .half()
                    )
                    expected_k_pool = (
                        gathered_k.view(1, 2, -1, query_block, 128)
                        .float()
                        .sum(3)
                        .div(denominator)
                        .half()
                    )
                    # Q128 streams two K64 halves through one CTA, so FP32
                    # accumulation order may move the final FP16 mean by one ULP.
                    torch.testing.assert_close(q_pool, expected_q_pool, atol=7e-5, rtol=0)
                    torch.testing.assert_close(k_pool, expected_k_pool, atol=7e-5, rtol=0)

                    expected_q8, expected_q_scale = _quantize_reference(
                        gathered_q, query_block
                    )
                    expected_k8, expected_k_scale = _quantize_reference(expected_k, 64)
                    self.assertTrue(torch.equal(q8, expected_q8))
                    self.assertTrue(torch.equal(k8, expected_k8))
                    torch.testing.assert_close(q_scale, expected_q_scale, atol=1e-8, rtol=0)
                    torch.testing.assert_close(k_scale, expected_k_scale, atol=1e-8, rtol=0)

    @torch.inference_mode()
    def test_maxpool_descriptors_and_native_probability_fusion(self) -> None:
        torch.manual_seed(23)
        prefix_tokens = 70
        video_tokens = 134
        for dtype in (torch.float16, torch.bfloat16):
            for query_block in (64, 128):
                with self.subTest(dtype=dtype, query_block=query_block):
                    indices, valid, counts = _layout(query_block)
                    shape = (1, 2, prefix_tokens + video_tokens, 128)
                    query = torch.randn(shape, device="cuda").to(dtype)
                    key = torch.randn(shape, device="cuda").to(dtype)
                    value = torch.randn(shape, device="cuda").to(dtype)
                    prepared = prepare_h3_sm89_int8_operands(
                        query,
                        key,
                        value,
                        indices,
                        valid,
                        counts,
                        prefix_tokens=prefix_tokens,
                        query_block_size=query_block,
                        smooth_k=False,
                        has_maxpool=True,
                    )
                    q_mean, k_mean = prepared[:2]
                    q_max, k_max = prepared[12:14]
                    mask = valid.view(1, 1, -1, query_block, 1)
                    for source, actual in ((query, q_max), (key, k_max)):
                        gathered = source[:, :, prefix_tokens + indices].half()
                        expected = (
                            gathered.view(1, 2, -1, query_block, 128)
                            .masked_fill(~mask, -torch.inf)
                            .amax(3)
                        )
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

                    def reference(
                        q_pool: torch.Tensor, k_pool: torch.Tensor
                    ) -> torch.Tensor:
                        rows = q_pool.size(2)
                        logits = torch.baddbmm(
                            torch.zeros(
                                (q_pool.size(0) * q_pool.size(1), rows, rows),
                                device="cuda",
                                dtype=torch.float16,
                            ),
                            q_pool.flatten(0, 1),
                            k_pool.flatten(0, 1).transpose(-1, -2),
                            beta=0,
                            alpha=1 / math.sqrt(128),
                        )
                        return logits.float().softmax(-1).half().view_as(
                            q_pool[..., :rows]
                        )

                    mean_probability = reference(q_mean, k_mean)
                    max_probability = reference(q_max, k_max)
                    for weight in (0.0, 0.1, 0.5, 1.0):
                        actual = sm89_h3_draft_probability(
                            q_mean,
                            k_mean,
                            q_max,
                            k_max,
                            maxpool_weight=weight,
                        )
                        expected = (
                            (1.0 - weight) * mean_probability.float()
                            + weight * max_probability.float()
                        ).half()
                        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=0)

    @torch.inference_mode()
    def test_k_smooth_centers_only_valid_rows_and_restores_pure_lse(self) -> None:
        torch.manual_seed(29)
        prefix_tokens = 70
        video_tokens = 134
        for query_block in (64, 128):
            with self.subTest(query_block=query_block):
                indices, valid, counts = _layout(query_block)
                shape = (1, 1, prefix_tokens + video_tokens, 128)
                query = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.2
                key = torch.randn_like(query) * 0.2
                value = torch.randn_like(query) * 0.2
                prepared = prepare_h3_sm89_int8_operands(
                    query,
                    key,
                    value,
                    indices,
                    valid,
                    counts,
                    prefix_tokens=prefix_tokens,
                    query_block_size=query_block,
                    smooth_k=True,
                )
                packed_q, packed_k = prepared[2], prepared[3]
                q8, k8, v8 = prepared[5:8]
                q_scale, k_scale, v_scale, key_mean = prepared[8:12]
                torch.testing.assert_close(
                    key_mean, key.float().mean(2).half(), atol=0, rtol=0
                )

                prefix_capacity = math.ceil(prefix_tokens / 64) * 64
                row_valid = torch.cat(
                    (
                        torch.arange(prefix_capacity, device="cuda") < prefix_tokens,
                        valid,
                    )
                ).view(1, 1, -1, 1)
                centered = torch.where(
                    row_valid,
                    (packed_k.float() - key_mean.float().unsqueeze(2)).half(),
                    0.0,
                )
                expected_k8, expected_k_scale = _quantize_reference(centered, 64)
                self.assertTrue(torch.equal(k8, expected_k8))
                torch.testing.assert_close(k_scale, expected_k_scale, atol=1e-8, rtol=0)
                self.assertEqual(
                    k8.masked_select(~row_valid.expand_as(k8)).count_nonzero().item(),
                    0,
                )

                key_blocks = packed_k.size(2) // 64
                query_blocks = packed_q.size(2) // query_block
                route = (
                    torch.arange(key_blocks, device="cuda", dtype=torch.int32)
                    .view(1, 1, 1, key_blocks)
                    .expand(1, 1, query_blocks, key_blocks)
                    .contiguous()
                )
                route_counts = torch.full(
                    (1, 1, query_blocks),
                    key_blocks,
                    device="cuda",
                    dtype=torch.int32,
                )
                prefix_counts = torch.tensor(
                    [64, 6], device="cuda", dtype=torch.int32
                )
                video_counts = counts.view(1, -1)
                if query_block == 128:
                    halves = torch.stack(
                        (
                            video_counts.clamp_max(64),
                            (video_counts - 64).clamp_min(0),
                        ),
                        dim=-1,
                    ).flatten()
                else:
                    halves = video_counts.flatten()
                valid_k = torch.cat((prefix_counts, halves)).view(1, -1)
                _, lse = native_k64_mixed_attention(
                    packed_q,
                    packed_k,
                    prepared[4],
                    route,
                    route_counts,
                    route,
                    torch.zeros_like(route_counts),
                    valid_k,
                    prepared_operands=(
                        q8,
                        k8,
                        v8,
                        q_scale,
                        k_scale,
                        v_scale,
                        key_mean,
                    ),
                    active_fp16=False,
                    query_block=query_block,
                )
                q_dequant = (
                    q8.float()
                    * q_scale.repeat_interleave(query_block, 2).unsqueeze(-1)
                )
                k_dequant = k8.float() * k_scale.repeat_interleave(64, 2).unsqueeze(-1)
                score = torch.matmul(q_dequant, k_dequant.transpose(-1, -2))
                score.add_(
                    torch.matmul(
                        packed_q.float(), key_mean.float().unsqueeze(-1)
                    )
                ).div_(math.sqrt(128))
                stage_valid = valid_k.flatten().repeat_interleave(64)
                positions = torch.arange(packed_k.size(2), device="cuda") % 64
                score.masked_fill_(
                    ~(positions < stage_valid).view(1, 1, 1, -1), -torch.inf
                )
                expected_lse = torch.logsumexp(score, dim=-1)
                torch.testing.assert_close(lse, expected_lse, atol=1.5e-5, rtol=0)

                low_blocks = key_blocks // 2
                low_counts = torch.full_like(route_counts, low_blocks)
                high_counts = torch.full_like(route_counts, key_blocks - low_blocks)
                output, mixed_lse = native_k64_mixed_attention(
                    packed_q,
                    packed_k,
                    prepared[4],
                    route,
                    low_counts,
                    route,
                    high_counts,
                    valid_k,
                    prepared_operands=(
                        q8,
                        k8,
                        v8,
                        q_scale,
                        k_scale,
                        v_scale,
                        key_mean,
                    ),
                    query_block=query_block,
                )
                low_tokens = low_blocks * 64
                low_score = torch.matmul(
                    q_dequant, k_dequant[:, :, :low_tokens].transpose(-1, -2)
                )
                low_score.add_(
                    torch.matmul(packed_q.float(), key_mean.float().unsqueeze(-1))
                )
                high_score = torch.matmul(
                    packed_q.float(), packed_k[:, :, low_tokens:].float().transpose(-1, -2)
                )
                mixed_score = torch.cat((low_score, high_score), dim=-1).div_(
                    math.sqrt(128)
                )
                mixed_score.masked_fill_(
                    ~(positions < stage_valid).view(1, 1, 1, -1), -torch.inf
                )
                expected_mixed_lse = torch.logsumexp(mixed_score, dim=-1)
                torch.testing.assert_close(
                    mixed_lse, expected_mixed_lse, atol=2.0e-3, rtol=2.0e-4
                )

                low_value = (
                    v8.float().transpose(-1, -2)
                    * v_scale.float().unsqueeze(2)
                )[:, :, :low_tokens]
                mixed_value = torch.cat(
                    (low_value, prepared[4][:, :, low_tokens:].float()), dim=2
                )
                expected_output = torch.matmul(
                    torch.softmax(mixed_score, dim=-1), mixed_value
                )
                torch.testing.assert_close(
                    output.float(), expected_output, atol=3.0e-2, rtol=3.0e-2
                )


if __name__ == "__main__":
    unittest.main()
