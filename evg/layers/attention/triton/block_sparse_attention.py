from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU dependency
    torch = None
    triton = None
    tl = None


@dataclass(frozen=True)
class TritonBlockSparseMetadata:
    full_indices: Any
    full_counts: Any
    partial_indices: Any
    partial_counts: Any


if triton is not None:  # pragma: no cover - exercised on CUDA/Triton machines

    @triton.jit
    def _compact_block_mask_kernel(
        mask_ptr,
        indices_ptr,
        counts_ptr,
        k_blocks: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        in_bounds = offsets < k_blocks
        selected = tl.load(
            mask_ptr + row * k_blocks + offsets, mask=in_bounds, other=0
        ).to(tl.int32)
        positions = tl.cumsum(selected, axis=0) - 1
        tl.store(
            indices_ptr + row * k_blocks + positions,
            offsets,
            mask=in_bounds & (selected != 0),
        )
        tl.store(counts_ptr + row, tl.sum(selected, axis=0))

    @triton.jit
    def _attend_indexed_tiles(
        q,
        k_ptr,
        v_ptr,
        block_indices_ptr,
        selected_count,
        token_mask_ptr,
        acc,
        l_i,
        m_i,
        batch,
        head,
        rows,
        cols_in_tile,
        dims,
        stride_kb: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        seq_k: tl.constexpr,
        qk_scale_log2: tl.constexpr,
        causal: tl.constexpr,
        LOGICAL_BLOCK_N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        APPLY_TOKEN_MASK: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        DISALLOW_ACC_MULTI_BUFFER: tl.constexpr,
        LOOP_UNROLL: tl.constexpr,
    ):
        sub_blocks = LOGICAL_BLOCK_N // BLOCK_N
        selected_tiles = selected_count * sub_blocks
        for tile_pos in tl.range(
            0,
            selected_tiles,
            num_stages=PIPE_STAGES,
            disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER,
            loop_unroll_factor=LOOP_UNROLL,
        ):
            selected_pos = tile_pos // sub_blocks
            sub_block = tile_pos % sub_blocks
            logical_k_block = tl.load(block_indices_ptr + selected_pos)
            cols = (
                logical_k_block * LOGICAL_BLOCK_N
                + sub_block * BLOCK_N
                + cols_in_tile
            )
            key_is_valid = cols < seq_k
            if APPLY_TOKEN_MASK:
                key_is_valid = key_is_valid & tl.load(
                    token_mask_ptr + batch * seq_k + cols,
                    mask=cols < seq_k,
                    other=0,
                ).to(tl.int1)
                has_valid_key = tl.sum(key_is_valid.to(tl.int32), axis=0) > 0
            else:
                has_valid_key = True

            k_offsets = (
                batch * stride_kb
                + cols[None, :] * stride_ks
                + head * stride_kh
                + dims[:, None] * stride_kd
            )
            key = tl.load(
                k_ptr + k_offsets,
                mask=key_is_valid[None, :],
                other=0.0,
            )
            scores = tl.dot(q, key) * qk_scale_log2
            score_mask = key_is_valid[None, :]
            if causal:
                score_mask = score_mask & (cols[None, :] <= rows[:, None])
            scores = tl.where(score_mask, scores, -float("inf"))

            block_m = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, block_m)
            safe_m_new = tl.where(has_valid_key, m_new, 0.0)
            alpha = tl.where(
                has_valid_key, tl.exp2(m_i - safe_m_new), 1.0
            )
            probabilities = tl.where(
                score_mask, tl.exp2(scores - safe_m_new[:, None]), 0.0
            )
            l_i = l_i * alpha + tl.sum(probabilities, axis=1)

            v_offsets = (
                batch * stride_vb
                + cols[:, None] * stride_vs
                + head * stride_vh
                + dims[None, :] * stride_vd
            )
            value = tl.load(
                v_ptr + v_offsets,
                mask=key_is_valid[:, None],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(
                probabilities.to(value.dtype), value
            )
            m_i = tl.where(has_valid_key, m_new, m_i)

        return acc, l_i, m_i

    @triton.jit
    def _indexed_block_sparse_attn_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        full_indices_ptr,
        full_counts_ptr,
        partial_indices_ptr,
        partial_counts_ptr,
        token_mask_ptr,
        out_ptr,
        stride_qb: tl.constexpr,
        stride_qs: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_os: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        seq_q: tl.constexpr,
        seq_k: tl.constexpr,
        heads: tl.constexpr,
        q_blocks: tl.constexpr,
        k_blocks: tl.constexpr,
        qk_scale_log2: tl.constexpr,
        causal: tl.constexpr,
        LOGICAL_BLOCK_M: tl.constexpr,
        LOGICAL_BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HAS_TOKEN_MASK: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        DISALLOW_ACC_MULTI_BUFFER: tl.constexpr,
        LOOP_UNROLL: tl.constexpr,
    ):
        query_tile = tl.program_id(0)
        head = tl.program_id(1)
        batch = tl.program_id(2)

        rows = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        cols_in_tile = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, HEAD_DIM)
        query_in_bounds = rows < seq_q
        q_offsets = (
            batch * stride_qb
            + rows[:, None] * stride_qs
            + head * stride_qh
            + dims[None, :] * stride_qd
        )
        q = tl.load(q_ptr + q_offsets, mask=query_in_bounds[:, None], other=0.0)

        logical_query_block = query_tile // (LOGICAL_BLOCK_M // BLOCK_M)
        row_id = (batch * heads + head) * q_blocks + logical_query_block
        full_count = tl.load(full_counts_ptr + row_id)
        partial_count = tl.load(partial_counts_ptr + row_id)
        full_row_ptr = full_indices_ptr + row_id * k_blocks
        partial_row_ptr = partial_indices_ptr + row_id * k_blocks
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        acc, l_i, m_i = _attend_indexed_tiles(
            q,
            k_ptr,
            v_ptr,
            full_row_ptr,
            full_count,
            token_mask_ptr,
            acc,
            l_i,
            m_i,
            batch,
            head,
            rows,
            cols_in_tile,
            dims,
            stride_kb,
            stride_ks,
            stride_kh,
            stride_kd,
            stride_vb,
            stride_vs,
            stride_vh,
            stride_vd,
            seq_k,
            qk_scale_log2,
            causal,
            LOGICAL_BLOCK_N,
            BLOCK_N,
            False,
            PIPE_STAGES,
            DISALLOW_ACC_MULTI_BUFFER,
            LOOP_UNROLL,
        )
        if HAS_TOKEN_MASK:
            acc, l_i, m_i = _attend_indexed_tiles(
                q,
                k_ptr,
                v_ptr,
                partial_row_ptr,
                partial_count,
                token_mask_ptr,
                acc,
                l_i,
                m_i,
                batch,
                head,
                rows,
                cols_in_tile,
                dims,
                stride_kb,
                stride_ks,
                stride_kh,
                stride_kd,
                stride_vb,
                stride_vs,
                stride_vh,
                stride_vd,
                seq_k,
                qk_scale_log2,
                causal,
                LOGICAL_BLOCK_N,
                BLOCK_N,
                True,
                PIPE_STAGES,
                DISALLOW_ACC_MULTI_BUFFER,
                LOOP_UNROLL,
            )

        output = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
        if HAS_TOKEN_MASK:
            query_is_valid = tl.load(
                token_mask_ptr + batch * seq_q + rows,
                mask=query_in_bounds,
                other=0,
            ).to(tl.int1)
            output = tl.where(query_is_valid[:, None], output, 0.0)

        out_offsets = (
            batch * stride_ob
            + rows[:, None] * stride_os
            + head * stride_oh
            + dims[None, :] * stride_od
        )
        tl.store(out_ptr + out_offsets, output, mask=query_in_bounds[:, None])

    @triton.jit
    def _indexed_block_sparse_split_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        full_indices_ptr,
        full_counts_ptr,
        partial_indices_ptr,
        partial_counts_ptr,
        token_mask_ptr,
        partial_out_ptr,
        partial_lse_ptr,
        stride_qb: tl.constexpr,
        stride_qs: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ps: tl.constexpr,
        stride_pb: tl.constexpr,
        stride_pt: tl.constexpr,
        stride_ph: tl.constexpr,
        stride_pd: tl.constexpr,
        stride_ls: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lt: tl.constexpr,
        stride_lh: tl.constexpr,
        seq_q: tl.constexpr,
        seq_k: tl.constexpr,
        heads: tl.constexpr,
        q_blocks: tl.constexpr,
        k_blocks: tl.constexpr,
        dense_q_start_block: tl.constexpr,
        qk_scale_log2: tl.constexpr,
        causal: tl.constexpr,
        LOGICAL_BLOCK_M: tl.constexpr,
        LOGICAL_BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HAS_TOKEN_MASK: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        KV_SPLITS: tl.constexpr,
    ):
        dense_query_block = tl.program_id(0)
        head = tl.program_id(1)
        batch_split = tl.program_id(2)
        batch = batch_split // KV_SPLITS
        split = batch_split % KV_SPLITS

        relative_rows = dense_query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        rows = (
            (dense_q_start_block + dense_query_block) * LOGICAL_BLOCK_M
            + tl.arange(0, BLOCK_M)
        )
        cols_in_tile = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, HEAD_DIM)
        query_in_bounds = rows < seq_q
        q_offsets = (
            batch * stride_qb
            + rows[:, None] * stride_qs
            + head * stride_qh
            + dims[None, :] * stride_qd
        )
        q = tl.load(q_ptr + q_offsets, mask=query_in_bounds[:, None], other=0.0)

        logical_query_block = dense_q_start_block + dense_query_block
        row_id = (batch * heads + head) * q_blocks + logical_query_block
        full_count = tl.load(full_counts_ptr + row_id)
        partial_count = tl.load(partial_counts_ptr + row_id)
        full_start = full_count * split // KV_SPLITS
        full_end = full_count * (split + 1) // KV_SPLITS
        partial_start = partial_count * split // KV_SPLITS
        partial_end = partial_count * (split + 1) // KV_SPLITS
        full_row_ptr = full_indices_ptr + row_id * k_blocks + full_start
        partial_row_ptr = partial_indices_ptr + row_id * k_blocks + partial_start

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        acc, l_i, m_i = _attend_indexed_tiles(
            q,
            k_ptr,
            v_ptr,
            full_row_ptr,
            full_end - full_start,
            token_mask_ptr,
            acc,
            l_i,
            m_i,
            batch,
            head,
            rows,
            cols_in_tile,
            dims,
            stride_kb,
            stride_ks,
            stride_kh,
            stride_kd,
            stride_vb,
            stride_vs,
            stride_vh,
            stride_vd,
            seq_k,
            qk_scale_log2,
            causal,
            LOGICAL_BLOCK_N,
            BLOCK_N,
            False,
            PIPE_STAGES,
            False,
            1,
        )
        if HAS_TOKEN_MASK:
            acc, l_i, m_i = _attend_indexed_tiles(
                q,
                k_ptr,
                v_ptr,
                partial_row_ptr,
                partial_end - partial_start,
                token_mask_ptr,
                acc,
                l_i,
                m_i,
                batch,
                head,
                rows,
                cols_in_tile,
                dims,
                stride_kb,
                stride_ks,
                stride_kh,
                stride_kd,
                stride_vb,
                stride_vs,
                stride_vh,
                stride_vd,
                seq_k,
                qk_scale_log2,
                causal,
                LOGICAL_BLOCK_N,
                BLOCK_N,
                True,
                PIPE_STAGES,
                False,
                1,
            )

        output = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
        lse = tl.where(l_i > 0.0, m_i + tl.log2(l_i), -float("inf"))
        if HAS_TOKEN_MASK:
            query_is_valid = tl.load(
                token_mask_ptr + batch * seq_q + rows,
                mask=query_in_bounds,
                other=0,
            ).to(tl.int1)
            output = tl.where(query_is_valid[:, None], output, 0.0)
            lse = tl.where(query_is_valid, lse, -float("inf"))

        partial_offsets = (
            split * stride_ps
            + batch * stride_pb
            + relative_rows[:, None] * stride_pt
            + head * stride_ph
            + dims[None, :] * stride_pd
        )
        lse_offsets = (
            split * stride_ls
            + batch * stride_lb
            + relative_rows * stride_lt
            + head * stride_lh
        )
        tl.store(
            partial_out_ptr + partial_offsets,
            output,
            mask=query_in_bounds[:, None],
        )
        tl.store(partial_lse_ptr + lse_offsets, lse, mask=query_in_bounds)

    @triton.jit
    def _merge_split_attention_kernel(
        partial_out_ptr,
        partial_lse_ptr,
        out_ptr,
        stride_ps: tl.constexpr,
        stride_pb: tl.constexpr,
        stride_pt: tl.constexpr,
        stride_ph: tl.constexpr,
        stride_pd: tl.constexpr,
        stride_ls: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lt: tl.constexpr,
        stride_lh: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_os: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        seq_q: tl.constexpr,
        dense_q_start_block: tl.constexpr,
        LOGICAL_BLOCK_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KV_SPLITS: tl.constexpr,
    ):
        dense_query_block = tl.program_id(0)
        head = tl.program_id(1)
        batch = tl.program_id(2)
        relative_rows = dense_query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        rows = (
            (dense_q_start_block + dense_query_block) * LOGICAL_BLOCK_M
            + tl.arange(0, BLOCK_M)
        )
        dims = tl.arange(0, HEAD_DIM)
        query_in_bounds = rows < seq_q

        global_m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        for split in range(0, KV_SPLITS):
            lse_offsets = (
                split * stride_ls
                + batch * stride_lb
                + relative_rows * stride_lt
                + head * stride_lh
            )
            split_lse = tl.load(
                partial_lse_ptr + lse_offsets,
                mask=query_in_bounds,
                other=-float("inf"),
            )
            global_m = tl.maximum(global_m, split_lse)

        safe_global_m = tl.where(global_m > -float("inf"), global_m, 0.0)
        denominator = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        for split in range(0, KV_SPLITS):
            lse_offsets = (
                split * stride_ls
                + batch * stride_lb
                + relative_rows * stride_lt
                + head * stride_lh
            )
            split_lse = tl.load(
                partial_lse_ptr + lse_offsets,
                mask=query_in_bounds,
                other=-float("inf"),
            )
            weight = tl.where(
                split_lse > -float("inf"),
                tl.exp2(split_lse - safe_global_m),
                0.0,
            )
            partial_offsets = (
                split * stride_ps
                + batch * stride_pb
                + relative_rows[:, None] * stride_pt
                + head * stride_ph
                + dims[None, :] * stride_pd
            )
            split_output = tl.load(
                partial_out_ptr + partial_offsets,
                mask=query_in_bounds[:, None],
                other=0.0,
            )
            denominator += weight
            acc += split_output * weight[:, None]

        output = tl.where(
            denominator[:, None] > 0.0,
            acc / denominator[:, None],
            0.0,
        )
        out_offsets = (
            batch * stride_ob
            + rows[:, None] * stride_os
            + head * stride_oh
            + dims[None, :] * stride_od
        )
        tl.store(out_ptr + out_offsets, output, mask=query_in_bounds[:, None])


def _next_power_of_2(value: int) -> int:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    return int(triton.next_power_of_2(value))


def _compact_block_mask(block_mask: Any) -> tuple[Any, Any]:
    rows = block_mask.numel() // block_mask.shape[-1]
    k_blocks = block_mask.shape[-1]
    indices = torch.empty_like(block_mask, dtype=torch.int32)
    counts = torch.empty((rows,), dtype=torch.int32, device=block_mask.device)
    _compact_block_mask_kernel[(rows,)](
        block_mask,
        indices,
        counts,
        k_blocks=k_blocks,
        BLOCK_K=_next_power_of_2(k_blocks),
        num_warps=4,
    )
    return indices, counts


def prepare_triton_block_sparse_metadata(
    block_mask: Any,
    token_mask: Any | None = None,
    k_block_size: int = 128,
) -> TritonBlockSparseMetadata:
    if triton is None or torch is None:
        raise ImportError("Triton is not installed")
    if not block_mask.is_cuda:
        raise RuntimeError("Triton block-mask compaction requires a CUDA tensor")

    block_mask = block_mask.to(dtype=torch.bool).contiguous()
    batch = block_mask.shape[0]
    k_blocks = block_mask.shape[-1]
    if token_mask is None:
        full_mask = block_mask
        partial_mask = torch.zeros_like(block_mask)
    else:
        seq_k = token_mask.shape[1]
        padded_len = k_blocks * k_block_size
        padded_token_mask = torch.nn.functional.pad(
            token_mask, (0, padded_len - seq_k), value=False
        )
        token_blocks = padded_token_mask.view(batch, k_blocks, k_block_size)
        full_k_blocks = token_blocks.all(dim=-1)
        partial_k_blocks = token_blocks.any(dim=-1) & ~full_k_blocks
        full_mask = block_mask & full_k_blocks[:, None, None, :]
        partial_mask = block_mask & partial_k_blocks[:, None, None, :]

    full_indices, full_counts = _compact_block_mask(full_mask.contiguous())
    partial_indices, partial_counts = _compact_block_mask(partial_mask.contiguous())
    return TritonBlockSparseMetadata(
        full_indices=full_indices,
        full_counts=full_counts,
        partial_indices=partial_indices,
        partial_counts=partial_counts,
    )


def _run_indexed_block_sparse_attention(
    q: Any,
    k: Any,
    v: Any,
    metadata: TritonBlockSparseMetadata,
    q_block_size: int,
    k_block_size: int,
    softmax_scale: float | None,
    causal: bool,
    token_mask: Any | None,
    block_m: int = 128,
    block_n: int = 64,
    num_warps: int = 8,
    num_stages: int = 3,
    disallow_acc_multi_buffer: bool = False,
    loop_unroll_factor: int = 1,
    dense_q_start_block: int | None = None,
    kv_splits: int = 2,
) -> Any:
    batch, seq_q, heads, head_dim = q.shape
    seq_k = k.shape[1]
    q_blocks = int(math.ceil(float(seq_q) / q_block_size))
    k_blocks = int(math.ceil(float(seq_k) / k_block_size))
    if dense_q_start_block is None:
        q_tiles = int(math.ceil(float(seq_q) / block_m))
    else:
        if q_block_size % block_m != 0:
            raise RuntimeError("q_block_size must be divisible by block_m")
        if not 0 <= dense_q_start_block <= q_blocks:
            raise ValueError("dense_q_start_block is outside the query block range")
        q_tiles = dense_q_start_block * (q_block_size // block_m)
    out = torch.empty_like(q)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    if q_tiles > 0:
        grid = (q_tiles, heads, batch)
        _indexed_block_sparse_attn_fwd_kernel[grid](
            q,
            k,
            v,
            metadata.full_indices,
            metadata.full_counts,
            metadata.partial_indices,
            metadata.partial_counts,
            token_mask if token_mask is not None else q,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            seq_q=seq_q,
            seq_k=seq_k,
            heads=heads,
            q_blocks=q_blocks,
            k_blocks=k_blocks,
            qk_scale_log2=scale * 1.4426950408889634,
            causal=causal,
            LOGICAL_BLOCK_M=q_block_size,
            LOGICAL_BLOCK_N=k_block_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim,
            HAS_TOKEN_MASK=token_mask is not None,
            PIPE_STAGES=num_stages,
            DISALLOW_ACC_MULTI_BUFFER=disallow_acc_multi_buffer,
            LOOP_UNROLL=loop_unroll_factor,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if dense_q_start_block is not None and dense_q_start_block < q_blocks:
        dense_q_blocks = q_blocks - dense_q_start_block
        dense_tokens = dense_q_blocks * q_block_size
        partial_out = torch.empty(
            (kv_splits, batch, dense_tokens, heads, head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        partial_lse = torch.empty(
            (kv_splits, batch, dense_tokens, heads),
            dtype=torch.float32,
            device=q.device,
        )
        split_grid = (dense_q_blocks, heads, batch * kv_splits)
        _indexed_block_sparse_split_fwd_kernel[split_grid](
            q,
            k,
            v,
            metadata.full_indices,
            metadata.full_counts,
            metadata.partial_indices,
            metadata.partial_counts,
            token_mask if token_mask is not None else q,
            partial_out,
            partial_lse,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *partial_out.stride(),
            *partial_lse.stride(),
            seq_q=seq_q,
            seq_k=seq_k,
            heads=heads,
            q_blocks=q_blocks,
            k_blocks=k_blocks,
            dense_q_start_block=dense_q_start_block,
            qk_scale_log2=scale * 1.4426950408889634,
            causal=causal,
            LOGICAL_BLOCK_M=q_block_size,
            LOGICAL_BLOCK_N=k_block_size,
            BLOCK_M=q_block_size,
            BLOCK_N=32,
            HEAD_DIM=head_dim,
            HAS_TOKEN_MASK=token_mask is not None,
            PIPE_STAGES=3,
            KV_SPLITS=kv_splits,
            num_warps=4,
            num_stages=3,
        )
        merge_grid = (dense_q_blocks, heads, batch)
        _merge_split_attention_kernel[merge_grid](
            partial_out,
            partial_lse,
            out,
            *partial_out.stride(),
            *partial_lse.stride(),
            *out.stride(),
            seq_q=seq_q,
            dense_q_start_block=dense_q_start_block,
            LOGICAL_BLOCK_M=q_block_size,
            BLOCK_M=q_block_size,
            HEAD_DIM=head_dim,
            KV_SPLITS=kv_splits,
            num_warps=4,
            num_stages=2,
        )
    return out


def triton_block_sparse_attention(
    q: Any,
    k: Any,
    v: Any,
    block_mask: Any,
    q_block_size: int,
    k_block_size: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    token_mask: Any | None = None,
    dense_q_start_block: int | None = None,
    kv_splits: int = 2,
) -> Any:
    if triton is None or torch is None:
        raise ImportError("Triton is not installed")
    if not q.is_cuda:
        raise RuntimeError("Triton block sparse attention requires CUDA tensors")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, and v must have shape [batch, seq, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("q, k, and v must share batch, heads, and head_dim")
    if q_block_size != 128 or k_block_size != 128:
        raise RuntimeError("The optimized Triton sparse kernel currently requires 128-token blocks")

    batch, seq_q, heads, head_dim = q.shape
    seq_k = k.shape[1]
    if head_dim not in (64, 128):
        raise RuntimeError("The optimized Triton sparse kernel supports head_dim 64 or 128")
    q_blocks = int(math.ceil(float(seq_q) / q_block_size))
    k_blocks = int(math.ceil(float(seq_k) / k_block_size))
    expected_mask_shape = (batch, heads, q_blocks, k_blocks)
    if tuple(block_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"block_mask must have shape {expected_mask_shape}, got {tuple(block_mask.shape)}"
        )

    block_mask = block_mask.to(device=q.device, dtype=torch.bool).contiguous()
    if token_mask is not None:
        if seq_q != seq_k:
            raise RuntimeError("token_mask currently requires equal query and key sequence lengths")
        if tuple(token_mask.shape) != (batch, seq_k):
            raise ValueError(
                f"token_mask must have shape {(batch, seq_k)}, got {tuple(token_mask.shape)}"
            )
        token_mask = token_mask.to(device=q.device, dtype=torch.bool).contiguous()

    metadata = prepare_triton_block_sparse_metadata(
        block_mask, token_mask=token_mask, k_block_size=k_block_size
    )
    return _run_indexed_block_sparse_attention(
        q,
        k,
        v,
        metadata,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        softmax_scale=softmax_scale,
        causal=causal,
        token_mask=token_mask,
        block_n=64 if token_mask is not None else 32,
        num_warps=8 if token_mask is not None else 4,
        num_stages=3,
        dense_q_start_block=dense_q_start_block,
        kv_splits=kv_splits,
    )
