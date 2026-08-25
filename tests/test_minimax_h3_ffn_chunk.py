from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


GB10_ROOT = (
    Path(__file__).resolve().parents[1]
    / "anemoi/models/minimax_h3/solengine/models/minimax_h3/gb10_fp8"
)
sys.path.insert(0, str(GB10_ROOT))
fusion_install = importlib.import_module("fusion_install")


class FFNChunkTests(unittest.TestCase):
    def test_chunking_starts_only_above_the_five_second_sequence(self) -> None:
        self.assertEqual(fusion_install._ffn_chunk_tokens(38_247, 32_768), 38_247)
        self.assertEqual(fusion_install._ffn_chunk_tokens(73_923, 32_768), 32_768)

    def test_project_swiglu_preserves_sequence_order(self) -> None:
        calls: list[int] = []

        def project(value: torch.Tensor) -> torch.Tensor:
            calls.append(value.shape[-2])
            return torch.cat((value, value + 1), dim=-1)

        def swiglu(value: torch.Tensor) -> torch.Tensor:
            left, right = value.chunk(2, dim=-1)
            return left * right

        hidden = torch.arange(42, dtype=torch.float32).reshape(1, 7, 6)
        with patch.object(fusion_install, "fused_swiglu", swiglu):
            actual = fusion_install._project_swiglu(project, hidden, 3)
            expected = swiglu(project(hidden))

        self.assertEqual(calls, [3, 3, 1, 7])
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
