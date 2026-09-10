"""Native D64 parity on real Dense-captured inputs (also replays D128).

Set ANEMOI_SM120_D64_CAPTURE to the indexed Cosmos3 Dense capture and
ANEMOI_SM120_TEST_D to 64 (default) or 128. Channel/token slices are explicit
operator workloads; Dense is recomputed for each slice, never reused from the
original full-width model. No random reference tensors are generated.
"""

from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from anemoi.layers.attention.mpa.backends import sm120_q64, sm120_q128
from anemoi.layers.attention.mpa.executor import _run_sm120_phases, sm120_ragged_h3_attention


CAPTURE = Path(os.environ.get("ANEMOI_SM120_D64_CAPTURE", ""))
HEAD_DIM = int(os.environ.get("ANEMOI_SM120_TEST_D", "64"))
FAMILIES = {
    "fp16": (0., 0., 0., 1.),
    "int8": (0., 1., 0., 0.),
    "int8_fp16": (0., .5, 0., .5),
    "mxfp8": (0., 0., 1., 0.),
    "mxfp8_fp16": (0., 0., .5, .5),
    "nvfp4": (1., 0., 0., 0.),
    "nvfp4_fp16": (.5, 0., 0., .5),
    "nvfp4_int8": (.5, .5, 0., 0.),
    "nvfp4_int8_fp16": (.34, .33, 0., .33),
    "nvfp4_mxfp8": (.5, 0., .5, 0.),
    "nvfp4_mxfp8_fp16": (.34, 0., .33, .33),
}


def captured_inputs(head_dim=HEAD_DIM, tokens=197, dtype=torch.float16):
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    assert payload["complete"] and payload["source"] == "cosmos3_dense_inference"
    return tuple(
        payload[name][:tokens, :heads, :head_dim].permute(1, 0, 2)
        .unsqueeze(0).to(device="cuda", dtype=dtype).contiguous()
        for name, heads in (("q_gen", 4), ("k_gen", 2), ("v_gen", 2))
    )


def prepare_case(query_block, *, head_dim=HEAD_DIM, tokens=197, prefix=0,
                 dtype=torch.float16):
    q, k, v = captured_inputs(head_dim, tokens, dtype)
    # The H3 producer contract has matching heads. Collapse the repeated KV
    # operands afterwards to exercise the native attention GQA consumer.
    k, v = (x.repeat_interleave(2, dim=1) for x in (k, v))
    video_tokens = tokens - prefix
    capacity = math.ceil(video_tokens / query_block) * query_block
    slots = torch.arange(capacity, device="cuda")
    valid = slots < video_tokens
    indices = slots.clamp_max(video_tokens - 1)
    counts = valid.view(-1, query_block).sum(-1).int()
    scales = tuple(torch.ones((), device="cuda") for _ in range(3))
    prepared = sm120_q64.prepare_h3_sm120_operands(
        q, k, v, indices, valid, counts, prefix_tokens=prefix,
        query_block_size=query_block, has_nvfp4=True, has_int8=True,
        has_mxfp8=True, has_fp16=True, has_prefix_query_int8=prefix > 0,
        has_maxpool=True, global_scales=scales,
    )
    return prepared, (q, k, v), indices, valid, counts, scales


def phase_call(prepared, scales, query_block, family, valid_k, *, empty=False,
               retained=None):
    ratios = FAMILIES[family]
    q, k, v = prepared[2:5]
    k, v = (x[:, ::2].contiguous() for x in (k, v))
    stages = k.size(2) // 64
    rows = q.size(2) // query_block
    ids = torch.arange(stages, device=q.device, dtype=torch.int32)
    ids = ids.view(1, 1, 1, -1).expand(1, q.size(1), rows, -1).contiguous()
    active = [i for i, ratio in enumerate(ratios) if ratio]
    keep = stages if retained is None else retained
    phase_stages = [0, 0, 0, 0]
    for i in active[:-1]:
        phase_stages[i] = max(1, int(keep * ratios[i]))
    phase_stages[active[-1]] = keep - sum(phase_stages)
    assert min(phase_stages) >= 0
    counts = [torch.full(ids.shape[:-1], n, device=q.device, dtype=torch.int32)
              for n in (phase_stages[0], phase_stages[1] + phase_stages[2], phase_stages[3])]
    if empty:
        for count in counts:
            count[:, :, -1] = 0

    def kv_operands(start):
        return tuple(x if i < 2 else x[:, ::2].contiguous()
                     for i, x in enumerate(prepared[start:start + 6]))

    kwargs = dict(
        query_block_size=query_block, ratios=ratios, query_fp16=q,
        key_fp16=k, value_fp16=v, block_ids=ids, nvfp4_counts=counts[0],
        middle_counts=counts[1], fp16_counts=(counts[2] if ratios[3] else
            torch.empty(0, device=q.device, dtype=torch.int32)), valid_k_counts=valid_k,
        layer=0, fp16_prefix_blocks=0, prepared_nv_operands=kv_operands(5),
        prepared_mxfp8_operands=kv_operands(11),
        prepared_int8_operands=kv_operands(17), prepared_global_scales=scales,
    )
    return lambda: _run_sm120_phases(**kwargs), kwargs, counts


def dense_reference(q, k, v):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), enable_gqa=True)


def quality(output, dense):
    actual, reference = output.float().flatten(), dense.float().flatten()
    return dict(relative_l2=float((actual - reference).norm() / reference.norm()),
                cosine=float(F.cosine_similarity(actual, reference, dim=0)))


@unittest.skipUnless(CAPTURE.is_file() and torch.cuda.is_available()
                     and torch.cuda.get_device_capability() == (12, 0),
                     "requires indexed Dense capture and native SM120 extension")
class SM120D64CudaTests(unittest.TestCase):
    @torch.inference_mode()
    def test_preparation_layout_scales_padding_and_pooling(self):
        for block in (64, 128):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(block=block, dtype=dtype):
                    p, raw, indices, valid, counts, scales = prepare_case(
                        block, prefix=13, dtype=dtype)
                    d = HEAD_DIM
                    q, k, v = p[2:5]
                    for data, scale, source in ((p[5], p[6], q), (p[7], p[8], k)):
                        self.assertEqual(data.shape, (*source.shape[:-1], d // 2))
                        self.assertEqual(scale.shape, (*source.shape[:-1], d // 16))
                    for data, scale, source in ((p[11], p[12], q), (p[13], p[14], k)):
                        self.assertEqual(data.shape, source.shape)
                        self.assertEqual(scale.shape, (*source.shape[:-1], d // 32))
                    self.assertEqual(p[10].shape[-1], d * 4)
                    self.assertEqual(p[16].shape[-1], d * 2)
                    self.assertEqual(p[22].shape, (1, 4, d))
                    for i, source in enumerate(raw):
                        video = source[:, :, 13 + indices].half().masked_fill(
                            ~valid.view(1, 1, -1, 1), 0)
                        expected = video if i == 0 else torch.cat((
                            F.pad(source[:, :, :13].half(), (0, 0, 0, 51)), video), dim=2)
                        torch.testing.assert_close(p[2 + i], expected, atol=0, rtol=0)
                        if i < 2:
                            tiles = video.view(1, 4, -1, block, d)
                            mean = (tiles.float().sum(3) / counts.view(1, 1, -1, 1)).half()
                            maximum = tiles.masked_fill(
                                ~valid.view(1, 1, -1, block, 1), -torch.inf).amax(3)
                            torch.testing.assert_close(p[i], mean, atol=2e-3, rtol=1e-3)
                            torch.testing.assert_close(p[25 + i], maximum, atol=0, rtol=0)
                    # Independent Q/K quantization group checks, including zero padding.
                    for source, data, scale in ((q, p[11], p[12]), (k, p[13], p[14])):
                        groups = source.float().view(*source.shape[:-1], d // 32, 32)
                        amax = groups.abs().amax(-1)
                        exponent = torch.ceil(torch.log2(amax / 448.)).clamp(-127, 127)
                        expected_scale = torch.where(amax == 0, 0, exponent + 127).byte()
                        torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
                        dequant = torch.exp2(scale.float() - 127).repeat_interleave(32, -1)
                        expected_data = (source.float() / dequant).to(torch.float8_e4m3fn).view(torch.uint8)
                        torch.testing.assert_close(data, expected_data, atol=0, rtol=0)
                    for source, data, scale, permute in (
                        (q, p[5], p[6], False), (k, p[7], p[8], True)
                    ):
                        groups = source.float().view(*source.shape[:-1], d // 16, 16)
                        expected_scale = (groups.abs().amax(-1) / 6).to(torch.float8_e4m3fn)
                        dequant = expected_scale.float().repeat_interleave(16, -1)
                        normalized = torch.where(dequant != 0, source.float() / dequant, 0)
                        midpoints = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device="cuda")
                        codes = torch.bucketize(normalized.abs().contiguous(), midpoints)
                        ties = normalized.abs() == midpoints[codes.clamp_max(6)]
                        codes += (ties & (codes % 2 == 1)).long()
                        codes = codes.byte() | (torch.signbit(normalized).byte() << 3)
                        expected_data = codes[..., ::2] | (codes[..., 1::2] << 4)
                        expected_scale = expected_scale.view(torch.uint8)
                        if permute:
                            rows = torch.arange(source.size(2), device="cuda")
                            local = rows % 32
                            permutation = rows - local + (local // 8) * 2 + ((local % 8) // 2) * 8 + local % 2
                            expected_data = expected_data[:, :, permutation]
                            expected_scale = expected_scale[:, :, permutation]
                        torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
                        torch.testing.assert_close(data, expected_data, atol=0, rtol=0)
                    for source, data, scale, tile in ((q, p[17], p[18], block), (k, p[19], p[20], 64)):
                        groups = source.float().view(1, 4, -1, tile, d)
                        expected_scale = groups.abs().amax((-1, -2)) / 127 + 1e-7
                        torch.testing.assert_close(scale, expected_scale, atol=1e-8, rtol=1e-6)
                        normalized = groups / scale[..., None, None]
                        expected_data = (normalized + torch.where(normalized >= 0, .5, -.5)).to(torch.int8)
                        torch.testing.assert_close(data, expected_data.view_as(data), atol=0, rtol=0)
                    standalone_mx = sm120_q64.prepare_mxfp8(q, k, v)
                    standalone_nv = (sm120_q64.prepare_q64_nvfp4 if block == 64
                                     else sm120_q128.prepare_q128_nvfp4)(q, k, v, *scales)
                    for actual, expected in zip((*p[5:11], *p[11:17]),
                                                (*standalone_nv, *standalone_mx)):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.inference_mode()
    def test_all_phase_families_gqa_ragged_and_dense(self):
        for block in (64, 128):
            for tokens in (197, 256):
                p, raw, _, valid, _, scales = prepare_case(block, tokens=tokens)
                valid_k = valid.view(1, -1, 64).sum(-1).int()
                dense = dense_reference(raw[0], raw[1][:, ::2], raw[2][:, ::2])
                for family in FAMILIES:
                    with self.subTest(block=block, tokens=tokens, family=family):
                        call, _, _ = phase_call(p, scales, block, family, valid_k)
                        out, lse = call()
                        out = out[:, :, :tokens]
                        self.assertEqual(out.shape, raw[0].shape)
                        self.assertEqual(lse.numel(), 0)
                        self.assertTrue(torch.isfinite(out).all())
                        metrics = quality(out, dense)
                        if family == "fp16":
                            torch.testing.assert_close(out.float(), dense, atol=2e-3, rtol=2e-2)
                        else:
                            limit = .30 if "nvfp4" in family else .08
                            self.assertLess(metrics["relative_l2"], limit, metrics)
                            self.assertGreater(metrics["cosine"], .95 if "nvfp4" in family else .995, metrics)

    @torch.inference_mode()
    def test_unpadded_q64_fp16_query(self):
        # The public producer pads Q. This direct-operator contract also
        # accepts raw Q tails; invalid query rows are never stored.
        p, raw, _, valid, _, scales = prepare_case(64, tokens=197)
        valid_k = valid.view(1, -1, 64).sum(-1).int()
        _, kwargs, _ = phase_call(p, scales, 64, "fp16", valid_k)
        kwargs["query_fp16"] = raw[0]
        out, _ = _run_sm120_phases(**kwargs)
        dense = dense_reference(raw[0], raw[1][:, ::2], raw[2][:, ::2])
        torch.testing.assert_close(out.float(), dense, atol=2e-3, rtol=2e-2)

    @torch.inference_mode()
    def test_empty_route_assembly_and_allocator_lifecycle(self):
        from anemoi.layers.attention.mpa.backends.sm89_k64 import assemble_h3_k64_output
        for block in (64, 128):
            p, _, indices, valid, _, scales = prepare_case(block, tokens=256)
            valid_k = valid.view(1, -1, 64).sum(-1).int()
            for family in FAMILIES:
                call, _, counts = phase_call(p, scales, block, family, valid_k, empty=True)
                for _ in range(4):
                    churn = torch.full_like(p[2], torch.nan)
                    del churn
                    out, lse = call()
                    assembled = assemble_h3_k64_output(
                        p[2][:, :, :0], out, indices, route_counts=tuple(counts),
                        query_block_size=block)
                    self.assertTrue(torch.isfinite(assembled).all(), family)
                    self.assertEqual(torch.count_nonzero(assembled[:, -block:]), 0)
                    self.assertFalse(torch.signbit(assembled[:, -block:]).any())
                    self.assertEqual(lse.numel(), 0)
            # A long-lived mixed-phase replay detects history-dependent storage.
            for _ in range(128):
                out, _ = call()
            torch.cuda.synchronize()

    @torch.inference_mode()
    def test_prefix_int8_gqa_and_compact_mxfp8(self):
        from anemoi.layers.attention.mpa.backends.sm120_q64 import resolve_mixed_attention_operator
        for block, backend in ((64, sm120_q64), (128, sm120_q128)):
            p, raw, _, valid, _, scales = prepare_case(block, prefix=13)
            valid_k = torch.cat((torch.tensor([[13]], device="cuda", dtype=torch.int32),
                                 valid.view(1, -1, 64).sum(-1).int()), dim=1)
            operands = tuple(x if i < 2 else x[:, ::2].contiguous()
                             for i, x in enumerate(p[17:23]))
            output = getattr(backend, f"sm120_q{block}_prefix_int8_attention")(
                p[23], p[24], operands, valid_k, 13)
            dense = dense_reference(raw[0][:, :, :13], raw[1][:, ::2], raw[2][:, ::2])
            self.assertLess(quality(output[:, :, :13], dense)["relative_l2"], .08)
        p, raw, _, valid, _, scales = prepare_case(128)
        valid_k = valid.view(1, -1, 64).sum(-1).int()
        _, kwargs, _ = phase_call(p, scales, 128, "mxfp8", valid_k)
        compact = resolve_mixed_attention_operator("sm120_q128_mxfp8_compact_attention_forward")
        out, _ = compact(
            *kwargs["prepared_mxfp8_operands"], kwargs["query_fp16"],
            kwargs["key_fp16"], kwargs["value_fp16"], kwargs["block_ids"],
            kwargs["middle_counts"], torch.empty(0, device="cuda", dtype=torch.int32),
            valid_k, 0, 1 / math.sqrt(HEAD_DIM), False)
        dense = dense_reference(raw[0], raw[1][:, ::2], raw[2][:, ::2])
        self.assertLess(quality(out[:, :, :197], dense)["relative_l2"], .08)

    @torch.inference_mode()
    def test_public_and_internal_routing_prefix_precision(self):
        from anemoi import anemoi_attention, VisualLayout, SparseConfig, QuantConfig
        for block in (64, 128):
            for dtype in (torch.float16, torch.bfloat16):
                q, k, v = captured_inputs(tokens=321, dtype=dtype)
                inputs = tuple(x.permute(0, 2, 1, 3) for x in
                               (q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)))
                dense = dense_reference(q, k, v).transpose(1, 2)
                for prefix in (0, 65):
                    shape = (1, 1, 321 - prefix)
                    for prefix_kv in ("fp16", "int8", "nvfp4"):
                        for prefix_q in ("fp16", "int8"):
                            out = anemoi_attention(*inputs, layout=VisualLayout(shape, prefix), layer=0,
                                sparse_config=SparseConfig(query_block_size=block, sparsity_ratio=0.),
                                quant_config=QuantConfig(nvfp4_ratio=.3, int8_ratio=.4, fp16_ratio=.3,
                                                         prefix_kv_precision=prefix_kv,
                                                         prefix_query_precision=prefix_q))
                            self.assertEqual(out.dtype, dtype)
                            self.assertEqual(out.shape, inputs[0].shape)
                            self.assertTrue(torch.isfinite(out).all())
                            self.assertLess(quality(out, dense)["relative_l2"], .30)
                for family, ratios in FAMILIES.items():
                    for maxpool in (0., .5):
                        out = sm120_ragged_h3_attention(*inputs, prefix_tokens=65,
                            video_shape=(1, 16, 16), layer=0, sparsity_ratio=0.,
                            query_block_size=block, retained_nvfp4_ratio=ratios[0],
                            retained_int8_ratio=ratios[1], retained_mxfp8_ratio=ratios[2],
                            retained_fp16_ratio=ratios[3], prefix_query_precision="fp16",
                            prefix_kv_precision="auto", maxpool_weight=maxpool, enable_anchors="_" not in family)
                        self.assertTrue(torch.isfinite(out).all(), family)
                        self.assertLess(quality(out, dense)["relative_l2"],
                                        .30 if "nvfp4" in family else .08, family)
                # Exercise actual sparse route selection as well as all-kept
                # precision checks. Sparse error is measured only against Dense.
                out = anemoi_attention(*inputs, layout=VisualLayout((1, 1, 321)), layer=0,
                    sparse_config=SparseConfig(query_block_size=block, sparsity_ratio=.5),
                    quant_config=QuantConfig(nvfp4_ratio=.3, int8_ratio=.4, fp16_ratio=.3))
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(math.isfinite(quality(out, dense)["relative_l2"]))

    @torch.inference_mode()
    def test_k_tail_native_descriptors_and_routing(self):
        p, raw, _, _, counts, _ = prepare_case(64)
        q_pool, k_pool = p[0], p[1][:, ::2].contiguous()
        packed_k = p[3][:, ::2].contiguous()
        blocks = packed_k.view(1, 2, -1, 64, HEAD_DIM)
        distances = (blocks.float() - k_pool.float().unsqueeze(3)).square().sum(-1)
        distances.masked_fill_(torch.arange(64, device="cuda").view(1, 1, 1, 64)
                               >= counts.view(1, 1, -1, 1), -torch.inf)
        for rank in (1, 2):
            extremes = torch.gather(blocks, 3, distances.topk(rank, dim=-1).indices
                                    .unsqueeze(-1).expand(-1, -1, -1, -1, HEAD_DIM))
            descriptors = torch.cat((k_pool.unsqueeze(3), extremes), dim=3)
            descriptors = descriptors.flatten(2, 3).repeat_interleave(2, dim=1)
            logits = (q_pool @ descriptors.transpose(-1, -2) / math.sqrt(HEAD_DIM)).half()
            logits = logits.view(1, 4, counts.numel(), counts.numel(), rank + 1).float()
            means, tails = logits[..., 0], logits[..., 1:]
            bulk_count = counts.view(1, 1, 1, -1) - rank
            bulk = (counts.view(1, 1, 1, -1) * means - tails.sum(-1)) / bulk_count
            reference = torch.logsumexp(torch.cat((bulk.unsqueeze(-1) + bulk_count.log().unsqueeze(-1),
                                                   tails), dim=-1), dim=-1).softmax(-1).half()
            operation = getattr(sm120_q64, f"sm120_h3_k_tail_r{rank}_probability")
            actual = operation(q_pool, k_pool, packed_k, counts, 0)
            torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-3)
            out = sm120_ragged_h3_attention(*(x.transpose(1, 2) for x in raw), prefix_tokens=0,
                video_shape=(1, 1, 197), layer=0, sparsity_ratio=0., query_block_size=64,
                retained_nvfp4_ratio=0., retained_int8_ratio=1., retained_fp16_ratio=0.,
                draftmap_proxy=f"k_tail_r{rank}")
            dense = dense_reference(*raw).transpose(1, 2)
            self.assertLess(quality(out, dense)["relative_l2"], .08)


if __name__ == "__main__":
    unittest.main()
