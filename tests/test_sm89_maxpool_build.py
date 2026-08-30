from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SM89MaxPoolBuildTests(unittest.TestCase):
    def test_maxpool_is_compile_time_optional_in_single_load_preparation(self) -> None:
        source = (
            ROOT / "csrc/attention/cuda/sm89/raster_preprocess.cu"
        ).read_text()

        self.assertIn("bool HasMaxPool", source)
        self.assertIn("if constexpr (HasMaxPool)", source)
        self.assertIn("float pool_max = -CUDART_INF_F", source)
        self.assertIn("has_maxpool ? torch::empty_like(q_pool)", source)
        self.assertIn("launch_qk(std::false_type{}, std::false_type{})", source)
        self.assertNotIn("maximum=True", source)

    def test_native_probability_reuses_fused_sm120_gemm_softmax(self) -> None:
        setup = (ROOT / "setup.py").read_text()
        api = (ROOT / "csrc/attention/cuda/sm89/api.h").read_text()
        bindings = (ROOT / "csrc/attention/cuda/sm89/bindings.cpp").read_text()
        draft = (
            ROOT / "csrc/attention/cuda/sm120/h3_draft_probability.cu"
        ).read_text()

        self.assertIn(
            '_source(attention / "sm120" / "h3_draft_probability.cu")', setup
        )
        for registered in (api, bindings):
            self.assertIn("sm89_h3_draft_probability", registered)
        self.assertIn("h3_draft_probability_impl", draft)
        self.assertIn("sm89_h3_draft_probability", draft)
        self.assertIn("row_softmax_fusion_fp16_kernel", draft)
        self.assertIn("maxpool_weight == 0.0", draft)
        self.assertIn("maxpool_weight == 1.0", draft)
        self.assertIn("launch_draft_gemm(q_max_pool", draft)


if __name__ == "__main__":
    unittest.main()
