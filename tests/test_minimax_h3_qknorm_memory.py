from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


FUSIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "anemoi/models/minimax_h3/solengine/models/minimax_h3/optimized/fusions.py"
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QKNormMemoryTests(unittest.TestCase):
    def test_fused_qknorm_rope_reuses_discarded_projection_storage(self) -> None:
        spec = importlib.util.spec_from_file_location("h3_optimized_fusions", FUSIONS_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        batch, sequence, heads, head_dim, rotary_dim = 1, 17, 2, 128, 96
        source = (
            torch.arange(batch * sequence * heads * head_dim, device="cuda")
            .remainder(257)
            .sub(128)
            .div(64)
            .to(torch.bfloat16)
            .reshape(batch, sequence, heads, head_dim)
        )
        original = source.clone()
        weight = torch.linspace(0.75, 1.25, head_dim, device="cuda")
        angles = torch.arange(sequence * rotary_dim, device="cuda").reshape(
            sequence, rotary_dim
        ) / 1000
        cosine, sine = angles.cos(), angles.sin()

        actual = module.fused_qknorm_rope(
            source, weight, cosine, sine, 1.0e-5
        )

        variance = original.float().square().mean(dim=-1, keepdim=True)
        normalized = original.float() * torch.rsqrt(variance + 1.0e-5) * weight
        rotary = normalized[..., :rotary_dim]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        expected = torch.cat(
            (rotary * cosine[:, None] + rotated * sine[:, None], normalized[..., rotary_dim:]),
            dim=-1,
        ).to(torch.bfloat16)

        self.assertEqual(actual.data_ptr(), source.data_ptr())
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)


if __name__ == "__main__":
    unittest.main()
