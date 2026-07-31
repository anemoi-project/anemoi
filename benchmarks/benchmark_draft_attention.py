from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
from flash_attn import flash_attn_func

from evg.layers.attention.draft_attention import DraftAttention, DraftAttentionConfig
from evg.layers.attention.sparse_attention import block_sparse_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the complete dynamic Draft Attention path against FlashAttention."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--latent-t", type=int, default=9)
    parser.add_argument("--latent-h", type=int, default=45)
    parser.add_argument("--latent-w", type=int, default=80)
    parser.add_argument("--text-len", type=int, default=1985)
    parser.add_argument("--pool-h", type=int, default=8)
    parser.add_argument("--pool-w", type=int, default=16)
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--dense-step-ratio", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def benchmark(
    operation: Callable[[], torch.Tensor | tuple[object, ...]],
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0.0 <= args.sparsity < 1.0:
        raise ValueError("--sparsity must be in [0, 1)")
    if not 0.0 <= args.dense_step_ratio <= 1.0:
        raise ValueError("--dense-step-ratio must be in [0, 1]")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    visual_len = args.latent_t * args.latent_h * args.latent_w
    sequence_len = visual_len + args.text_len
    shape = (args.batch_size, sequence_len, args.heads, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    text_mask = torch.ones(
        (args.batch_size, sequence_len), device="cuda", dtype=torch.bool
    )

    attention = DraftAttention(
        DraftAttentionConfig(
            latent_h=args.latent_h,
            latent_w=args.latent_w,
            visual_len=visual_len,
            text_len=args.text_len,
            pool_h=args.pool_h,
            pool_w=args.pool_w,
            sparsity_ratio=args.sparsity,
            backend="triton",
        )
    )

    block_mask, debug = attention.build_sequence_block_mask(q, k)
    reorg_idx, restore_idx, layout_valid_mask = attention._indices(q.device)
    q_reorg = q.index_select(1, reorg_idx)
    k_reorg = k.index_select(1, reorg_idx)
    v_reorg = v.index_select(1, reorg_idx)
    token_mask = layout_valid_mask.expand(args.batch_size, -1)

    def dense() -> torch.Tensor:
        return flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

    def draft_map() -> tuple[object, ...]:
        return attention.build_sequence_block_mask(q, k, collect_debug=False)

    def reorder() -> torch.Tensor:
        return torch.stack(
            (
                q.index_select(1, reorg_idx),
                k.index_select(1, reorg_idx),
                v.index_select(1, reorg_idx),
            ),
            dim=0,
        )

    def sparse() -> torch.Tensor:
        return block_sparse_attention(
            q_reorg,
            k_reorg,
            v_reorg,
            block_mask=block_mask,
            q_block_size=debug.q_block_size,
            k_block_size=debug.k_block_size,
            backend="triton",
            token_mask=token_mask,
            dense_q_start_block=debug.draft_q_blocks,
        )

    sparse_output = sparse()

    def restore() -> torch.Tensor:
        return sparse_output.index_select(1, restore_idx)

    def full_draft() -> torch.Tensor:
        return attention(q, k, v, attn_mask=text_mask)

    cases = {
        "dense_flash": dense,
        "draft_map": draft_map,
        "reorder_qkv": reorder,
        "sparse_attention": sparse,
        "restore_output": restore,
        "full_draft": full_draft,
    }
    results = {
        name: benchmark(operation, args.warmup, args.iterations)
        for name, operation in cases.items()
    }

    dense_ms = results["dense_flash"][0]
    draft_ms = results["full_draft"][0]
    scheduled_ms = (
        args.dense_step_ratio * dense_ms
        + (1.0 - args.dense_step_ratio) * draft_ms
    )
    padded_sequence_len = len(reorg_idx)

    print("EVG dynamic Draft Attention benchmark")
    print(f"gpu={torch.cuda.get_device_name(0)} dtype=bf16")
    print(
        f"shape=B{args.batch_size} S{sequence_len} H{args.heads} D{args.head_dim} "
        f"visual={visual_len} ({args.latent_t}x{args.latent_h}x{args.latent_w}) "
        f"text={args.text_len} padded_S={padded_sequence_len}"
    )
    print(
        f"target_sparsity={args.sparsity:.1%} visual_density={debug.draft_density:.3%} "
        f"full_sequence_density={debug.sequence_density:.3%} block=128x128"
    )
    for name, (median_ms, minimum_ms) in results.items():
        print(f"{name:>18}: median={median_ms:8.3f} ms min={minimum_ms:8.3f} ms")
    print(f"full_sparse_step_speedup={dense_ms / draft_ms:.3f}x")
    print(
        f"scheduled_attention_speedup={dense_ms / scheduled_ms:.3f}x "
        f"({args.dense_step_ratio:.0%} dense steps)"
    )


if __name__ == "__main__":
    main()
