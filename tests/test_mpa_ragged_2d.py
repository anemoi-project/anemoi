from __future__ import annotations

import math
import random
import unittest

from evg.layers.attention.mpa.ragged_2d import make_ragged_2d_partition
from evg.layers.attention.mpa.routing import route_probability


class Ragged2DPartitionTests(unittest.TestCase):
    def _assert_partition(self, height: int, width: int, capacity: int) -> None:
        partition = make_ragged_2d_partition(height, width, capacity)
        self.assertEqual(partition.block_count, math.ceil(height * width / capacity))
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


class RaggedRoutingTests(unittest.TestCase):
    def test_anchors_are_fp16_without_expanding_global_budget(self) -> None:
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
        self.assertEqual(int(plan.fp8_counts.sum()), 4)
        self.assertEqual(int(plan.fp16_counts.sum()), 4)
        for row in range(rows):
            first = int(plan.fp8_counts[0, 0, row])
            count = int(plan.fp16_counts[0, 0, row])
            fp16_ids = plan.block_ids[0, 0, row, first : first + count]
            self.assertIn(row + 3, fp16_ids.tolist())

    def test_infeasible_sparse_budget_expands_only_to_fit_anchors(self) -> None:
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
        plan = route_probability(
            probability,
            anchors,
            anchor_count=anchor_count,
            prefix_blocks=0,
            sparsity_ratio=0.88,
            fp8_ratio=0.8,
            fp16_ratio=0.2,
        )
        self.assertEqual(int(plan.fp8_counts.sum()), 0)
        self.assertEqual(int(plan.fp16_counts.sum()), anchor_count)
        selected = torch.zeros_like(anchors)
        for row in range(rows):
            first = int(plan.fp8_counts[0, 0, row])
            count = int(plan.fp16_counts[0, 0, row])
            selected[row, plan.block_ids[0, 0, row, first : first + count].long()] = True
        self.assertTrue(torch.equal(selected, anchors))


if __name__ == "__main__":
    unittest.main()
