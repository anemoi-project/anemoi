import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from evg.layers.attention.draft_attention import (
    DraftAttention,
    DraftAttentionConfig,
    generate_padded_reorg_layout,
    generate_reorg_restore_indices,
)
from evg.layers.attention.draft_map import (
    blockwise_draft_mask,
    dense_draft_attention_map,
    headwise_topk_mask,
)
from evg.layers.attention.sparse_attention import (
    block_sparse_attention_reference,
    dense_attention_reference,
)


@unittest.skipIf(torch is None, "PyTorch is required")
class DraftAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_reorg_restore_indices_round_trip(self) -> None:
        reorg, restore = generate_reorg_restore_indices(
            pool_h=2,
            pool_w=4,
            latent_h=4,
            latent_w=8,
            visual_len=64,
            text_len=3,
        )
        x = torch.arange(67)
        round_trip = x[reorg][restore]
        self.assertTrue(torch.equal(round_trip, x))

    def test_padded_reorg_layout_round_trip(self) -> None:
        reorg, restore, valid, padded_visual_len = generate_padded_reorg_layout(
            pool_h=4,
            pool_w=4,
            latent_h=5,
            latent_w=6,
            visual_len=60,
            text_len=3,
        )
        x = torch.arange(63)
        padded = x[reorg]

        self.assertEqual(padded_visual_len, 128)
        self.assertEqual(sum(valid), 63)
        self.assertTrue(torch.equal(padded[restore], x))

    def test_padded_reorg_forms_128_token_spatial_blocks(self) -> None:
        pool_h, pool_w = 8, 16
        latent_h, latent_w = 45, 80
        frame_tokens = latent_h * latent_w
        visual_len = 2 * frame_tokens
        reorg, restore, valid, padded_visual_len = generate_padded_reorg_layout(
            pool_h=pool_h,
            pool_w=pool_w,
            latent_h=latent_h,
            latent_w=latent_w,
            visual_len=visual_len,
            text_len=3,
        )

        block_size = pool_h * pool_w
        tile_rows = (latent_h + pool_h - 1) // pool_h
        tile_cols = (latent_w + pool_w - 1) // pool_w
        blocks_per_frame = tile_rows * tile_cols
        self.assertEqual(block_size, 128)
        self.assertEqual(padded_visual_len, 2 * blocks_per_frame * block_size)

        for block_idx in range(padded_visual_len // block_size):
            frame_idx = block_idx // blocks_per_frame
            frame_block_idx = block_idx % blocks_per_frame
            expected_tile_row = frame_block_idx // tile_cols
            expected_tile_col = frame_block_idx % tile_cols
            start = block_idx * block_size
            for old_position, is_valid in zip(
                reorg[start : start + block_size],
                valid[start : start + block_size],
            ):
                if not is_valid:
                    continue
                actual_frame = old_position // frame_tokens
                frame_position = old_position % frame_tokens
                row, col = divmod(frame_position, latent_w)
                self.assertEqual(actual_frame, frame_idx)
                self.assertEqual(row // pool_h, expected_tile_row)
                self.assertEqual(col // pool_w, expected_tile_col)

        x = torch.arange(visual_len + 3)
        self.assertTrue(torch.equal(x[reorg][restore], x))

    def test_blockwise_draft_mask_matches_dense_mask(self) -> None:
        q = torch.randn(10, 3, 8)
        k = torch.randn(10, 3, 8)
        keep_ratio = 0.35

        dense_map = dense_draft_attention_map(q, k)
        dense_mask = headwise_topk_mask(dense_map, keep_ratio=keep_ratio)
        block_mask = blockwise_draft_mask(
            q,
            k,
            keep_ratio=keep_ratio,
            q_chunk_size=4,
            k_chunk_size=3,
        ).block_mask

        self.assertTrue(torch.equal(block_mask.cpu(), dense_mask.cpu()))

    def test_sparse_attention_matches_dense_when_mask_is_full(self) -> None:
        q = torch.randn(1, 9, 2, 8)
        k = torch.randn(1, 9, 2, 8)
        v = torch.randn(1, 9, 2, 8)
        block_mask = torch.ones(1, 2, 3, 3, dtype=torch.bool)

        sparse = block_sparse_attention_reference(
            q,
            k,
            v,
            block_mask=block_mask,
            q_block_size=3,
            k_block_size=3,
        )
        dense = dense_attention_reference(q, k, v)
        self.assertTrue(torch.allclose(sparse, dense, atol=1e-5, rtol=1e-5))

    def test_sparse_attention_honors_token_mask(self) -> None:
        q = torch.randn(1, 8, 2, 8)
        k = torch.randn(1, 8, 2, 8)
        v = torch.randn(1, 8, 2, 8)
        block_mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
        token_mask = torch.tensor([[True, True, True, True, True, True, False, False]])

        sparse = block_sparse_attention_reference(
            q,
            k,
            v,
            block_mask=block_mask,
            q_block_size=4,
            k_block_size=4,
            token_mask=token_mask,
        )
        dense = dense_attention_reference(q[:, :6], k[:, :6], v[:, :6])

        self.assertTrue(torch.allclose(sparse[:, :6], dense, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(sparse[:, 6:], torch.zeros_like(sparse[:, 6:])))

    def test_draft_attention_full_keep_matches_dense(self) -> None:
        q = torch.randn(1, 64, 2, 8)
        k = torch.randn(1, 64, 2, 8)
        v = torch.randn(1, 64, 2, 8)
        attention = DraftAttention(
            DraftAttentionConfig(
                latent_h=4,
                latent_w=8,
                visual_len=64,
                pool_h=2,
                pool_w=4,
                sparsity_ratio=0.0,
                backend="torch",
            )
        )

        draft = attention(q, k, v)
        dense = dense_attention_reference(q, k, v)
        self.assertTrue(torch.allclose(draft, dense, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
