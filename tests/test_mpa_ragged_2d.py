from __future__ import annotations

import hashlib
import json
import math
import random
import unittest

from anemoi.layers.attention.mpa.ragged_2d import _partition_cost, make_ragged_2d_partition
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.routing import route_probability


class Ragged2DPartitionTests(unittest.TestCase):
    def _blocks_sha256(self, blocks: tuple[tuple[int, ...], ...]) -> str:
        payload = json.dumps(blocks, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def _assert_partition(self, height: int, width: int, capacity: int) -> None:
        partition = make_ragged_2d_partition(height, width, capacity)
        block_count = math.ceil(height * width / capacity)
        self.assertEqual(partition.block_count, block_count)
        q, r = divmod(height * width, block_count)
        self.assertEqual(
            sorted(partition.counts),
            [q] * (block_count - r) + [q + 1] * r,
        )
        tokens = [token for block in partition.blocks for token in block]
        self.assertEqual(sorted(tokens), list(range(height * width)))
        self.assertGreater(min(partition.counts), 0)
        self.assertLessEqual(max(partition.counts), capacity)
        for block_id, block in enumerate(partition.blocks):
            self.assertTrue(all(partition.token_to_block[token] == block_id for token in block))
            remaining = {divmod(token, width) for token in block}
            frontier = {remaining.pop()}
            visited = set(frontier)
            while frontier:
                row, column = frontier.pop()
                for neighbor in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        visited.add(neighbor)
                        frontier.add(neighbor)
            self.assertEqual(len(visited), len(block))

    def test_representative_arbitrary_resolutions(self) -> None:
        for capacity in (64, 128):
            for height, width in (
                (1, 4097),
                (7, 9),
                (23, 41),
                (24, 40),
                (24, 42),
                (27, 45),
                (31, 53),
                (64, 65),
                (4097, 1),
            ):
                with self.subTest(height=height, width=width, capacity=capacity):
                    self._assert_partition(height, width, capacity)

    def test_seeded_resolution_matrix(self) -> None:
        generator = random.Random(20260815)
        for _ in range(40):
            self._assert_partition(
                generator.randint(1, 71),
                generator.randint(1, 89),
                generator.choice((64, 128)),
            )

    def test_24x42_reaches_capacity_lower_bound(self) -> None:
        partition = make_ragged_2d_partition(24, 42, 64)
        self.assertEqual(partition.block_count, 16)
        self.assertEqual(sum(partition.counts), 1008)

    def test_balanced_mass_contract_handles_nonzero_remainder(self) -> None:
        partition = make_ragged_2d_partition(7, 20, 64)
        self.assertEqual(sorted(partition.counts), [46, 47, 47])

    def test_h3_q64_partition_has_sixteen_equal_63_token_blocks(self) -> None:
        partition = make_ragged_2d_partition(24, 42, 64)
        self.assertEqual(partition.counts, (63,) * 16)
        self.assertEqual(
            tuple(
                (
                    min(token // 42 for token in block),
                    max(token // 42 for token in block),
                )
                for block in partition.blocks
            ),
            ((0, 8),) * 6 + ((9, 14),) * 4 + ((15, 23),) * 6,
        )
        self.assertEqual(_partition_cost(partition.blocks, 42)[:2], (520.0, 12.0))

    def test_h3_q128_partition_has_eight_equal_126_token_blocks(self) -> None:
        partition = make_ragged_2d_partition(
            24, 42, 128, include_adjacency=False
        )
        self.assertEqual(partition.counts, (126,) * 8)
        self.assertEqual(sum(partition.counts), 1008)
        self.assertEqual(
            self._blocks_sha256(partition.blocks),
            "110c3cb43e9887edc77afaebfaac29feefc9232b9beea27711dc2ac64d71d06c",
        )

    def test_balanced_regular_layouts_remain_byte_identical(self) -> None:
        expected = {
            (8, 8, 64): "bed7ce3a87c88a80e42bebb28d80ea1d414684177870eda040e6e4af9d9e60f2",
            (16, 16, 64): "797b760414bab5e0ddef088fcc536d6fd86433f0cbc0d08bb51bb86978fcfc4b",
            (24, 40, 64): "465b75da839dd0166793796a4e2ed9a950d3007a55e31b94a8df32e093c8152f",
            (16, 16, 128): "8bfd7ff9cd7ea1baf4c07eb514da757932cea8bcc84d3a4893d34325efef4090",
            (12, 32, 64): "47f7df182539bdfb7175d06d75b67dfeb79d48def654d974304d27627bcc3060",
        }
        for geometry, expected_hash in expected.items():
            with self.subTest(geometry=geometry):
                height, width, capacity = geometry
                partition = make_ragged_2d_partition(
                    height, width, capacity, include_adjacency=False
                )
                self.assertEqual(_partition_cost(partition.blocks, width)[1], 0.0)
                self.assertEqual(self._blocks_sha256(partition.blocks), expected_hash)

    def test_anchor_free_partition_omits_adjacency(self) -> None:
        partition = make_ragged_2d_partition(
            7, 9, 64, include_adjacency=False
        )
        self.assertIsNone(partition.adjacency)


class RaggedRoutingTests(unittest.TestCase):
    def test_three_phase_route_compacts_nv_middle_fp16(self) -> None:
        import torch

        probability = torch.zeros((1, 1, 4, 4), dtype=torch.float16)
        probability[0, 0, 0] = torch.tensor((7.0, 5.0, 6.0, 4.0))
        plan = route_probability(
            probability,
            None,
            anchor_count=0,
            prefix_blocks=0,
            sparsity_ratio=0.75,
            nvfp4_ratio=0.25,
            fp8_ratio=0.25,
            fp16_ratio=0.50,
        )
        self.assertEqual(plan.block_ids[0, 0, 0, :4].tolist(), [3, 1, 0, 2])
        self.assertEqual(int(plan.nvfp4_counts[0, 0, 0]), 1)
        self.assertEqual(int(plan.fp8_counts[0, 0, 0]), 1)
        self.assertEqual(int(plan.fp16_counts[0, 0, 0]), 2)

    def test_two_phase_call_remains_backward_compatible(self) -> None:
        import torch

        probability = torch.arange(16, dtype=torch.float16).view(1, 1, 4, 4)
        plan = route_probability(
            probability,
            None,
            anchor_count=0,
            prefix_blocks=0,
            sparsity_ratio=0.5,
            fp8_ratio=0.75,
            fp16_ratio=0.25,
        )
        self.assertEqual(int(plan.nvfp4_counts.sum()), 0)
        self.assertEqual(int((plan.fp8_counts + plan.fp16_counts).sum()), 8)

    def test_default_route_uses_nominal_budget_without_anchors(self) -> None:
        import torch

        probability = torch.arange(16, dtype=torch.float16).view(1, 1, 4, 4)
        plan = route_probability(
            probability,
            None,
            anchor_count=0,
            prefix_blocks=0,
            sparsity_ratio=0.5,
            fp8_ratio=0.75,
            fp16_ratio=0.25,
        )
        self.assertEqual(int(plan.fp8_counts.sum() + plan.fp16_counts.sum()), 8)

    def test_anchor_free_route_breaks_ties_by_row_major_id(self) -> None:
        import torch

        probability = torch.zeros((1, 1, 2, 2), dtype=torch.float16)
        plan = route_probability(
            probability,
            None,
            anchor_count=0,
            prefix_blocks=0,
            sparsity_ratio=0.5,
            fp8_ratio=0.5,
            fp16_ratio=0.5,
        )
        fp8_selected = []
        fp16_selected = []
        for row in range(2):
            fp8 = int(plan.fp8_counts[0, 0, row])
            fp16 = int(plan.fp16_counts[0, 0, row])
            fp8_selected.extend(
                (row, int(column))
                for column in plan.block_ids[0, 0, row, :fp8]
            )
            fp16_selected.extend(
                (row, int(column))
                for column in plan.block_ids[0, 0, row, fp8 : fp8 + fp16]
            )
        self.assertEqual(fp16_selected, [(0, 0)])
        self.assertEqual(fp8_selected, [(0, 1)])

    def test_default_layout_does_not_materialize_anchors(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        layout = materialize_ragged_2d_layout(
            torch.device("cuda"),
            frames=1,
            height=7,
            width=9,
            logical_block=64,
            enable_anchors=False,
        )
        self.assertIsNone(layout.anchors)
        self.assertIsNone(layout.anchor_ids)
        self.assertEqual(layout.anchor_count, 0)

    def test_anchor_layout_caches_compact_flat_ids(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        layout = materialize_ragged_2d_layout(
            torch.device("cuda"),
            frames=2,
            height=7,
            width=9,
            logical_block=64,
            enable_anchors=True,
        )

        self.assertEqual(layout.anchor_ids.dtype, torch.int32)
        self.assertEqual(layout.anchor_ids.tolist(), [0, 3])
        self.assertEqual(layout.anchor_count, layout.anchor_ids.numel())
        self.assertTrue(torch.all(layout.anchors.flatten()[layout.anchor_ids.long()]))

    def test_missing_anchors_replace_weakest_low_precision_edges(self) -> None:
        import torch

        rows = 4
        probability = torch.arange(rows * rows, dtype=torch.float16).view(1, 1, rows, rows)
        anchors = torch.eye(rows, dtype=torch.bool)
        plan = route_probability(
            probability,
            anchors,
            anchor_count=rows,
            prefix_blocks=3,
            sparsity_ratio=0.5,
            fp8_ratio=0.75,
            fp16_ratio=0.25,
        )
        self.assertEqual(int(plan.fp8_counts.sum()), 6)
        self.assertEqual(int(plan.fp16_counts.sum()), 2)
        precision = torch.zeros((rows, rows), dtype=torch.uint8)
        for row in range(rows):
            middle = int(plan.fp8_counts[0, 0, row])
            high = int(plan.fp16_counts[0, 0, row])
            ids = plan.block_ids[0, 0, row] - 3
            precision[row, ids[:middle].long()] = 2
            precision[row, ids[middle : middle + high].long()] = 3
        flat = precision.flatten()
        self.assertTrue(torch.all(precision[anchors] > 0))
        self.assertEqual(int(flat[0]), 2)
        self.assertEqual(int(flat[5]), 2)
        self.assertEqual(int(flat[10]), 2)
        self.assertEqual(int(flat[15]), 3)
        self.assertEqual(int(flat[8]), 0)
        self.assertEqual(int(flat[9]), 0)

    def test_infeasible_lowest_precision_budget_is_rejected(self) -> None:
        import torch

        rows = 3
        probability = torch.arange(rows * rows, dtype=torch.float16).view(1, 1, rows, rows)
        anchors = torch.tensor(
            (
                (True, True, False),
                (True, True, True),
                (False, True, True),
            ),
            dtype=torch.bool,
        )
        anchor_count = int(anchors.sum())
        with self.assertRaisesRegex(ValueError, "lowest-precision budget"):
            route_probability(
                probability,
                anchors,
                anchor_count=anchor_count,
                prefix_blocks=0,
                sparsity_ratio=0.88,
                fp8_ratio=0.8,
                fp16_ratio=0.2,
            )


if __name__ == "__main__":
    unittest.main()
