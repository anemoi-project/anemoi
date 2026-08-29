import os
import unittest
from importlib.util import find_spec
from pathlib import Path

import torch

import anemoi
from anemoi.models.minimax_h3 import H3MPAAttention, H3MPAConfig

_CAPTURE = Path(os.environ.get("ANEMOI_ATTENTION_CAPTURE", ""))
_EXTENSIONS_AVAILABLE = all(
    find_spec(name) is not None
    for name in (
        "anemoi.layers.attention.mpa._cuda_attention",
        "anemoi.layers.attention.mpa._cuda_sm120_q64",
    )
)
_PREFIX_TOKENS = 951
_VIDEO_SHAPE = (37, 24, 42)
_PRECISION_CASES = (
    ("nvfp4", (1.0, 0.0, 0.0)),
    ("int8", (0.0, 1.0, 0.0)),
    ("fp16", (0.0, 0.0, 1.0)),
    ("nvfp4_int8", (0.6, 0.4, 0.0)),
    ("nvfp4_fp16", (0.85, 0.0, 0.15)),
    ("int8_fp16", (0.0, 0.85, 0.15)),
    ("nvfp4_int8_fp16", (0.6, 0.25, 0.15)),
)


def _prefix_precisions(ratios: tuple[float, float, float]) -> tuple[str, str]:
    nvfp4, int8, _ = ratios
    if int8:
        return "int8", "int8"
    return ("nvfp4", "fp16") if nvfp4 else ("fp16", "fp16")


def _release_smoke_ready(
    capture_available: bool,
    cuda_available: bool,
    capability: tuple[int, int] | None,
    extensions_available: bool,
) -> bool:
    if not capture_available or not cuda_available or capability != (12, 0):
        return False
    if not extensions_available:
        raise RuntimeError("release attention smoke requires both native extensions")
    return True


class AttentionAPICudaAvailabilityTests(unittest.TestCase):
    def test_ordinary_missing_requirements_skip(self) -> None:
        cases = (
            (False, True, (12, 0), False),
            (True, False, None, False),
            (True, True, (8, 9), False),
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(_release_smoke_ready(*case))

    def test_explicit_sm120_release_smoke_requires_extensions(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "native extensions"):
            _release_smoke_ready(True, True, (12, 0), False)
        self.assertTrue(_release_smoke_ready(True, True, (12, 0), True))


_CUDA_AVAILABLE = torch.cuda.is_available()
_CAPABILITY = torch.cuda.get_device_capability() if _CUDA_AVAILABLE else None


@unittest.skipUnless(
    _release_smoke_ready(
        _CAPTURE.is_file(),
        _CUDA_AVAILABLE,
        _CAPABILITY,
        _EXTENSIONS_AVAILABLE,
    ),
    "requires indexed Dense Q/K/V, native extensions, and SM120",
)
class AttentionAPICudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = torch.load(_CAPTURE, map_location="cpu", weights_only=True, mmap=True)
        if payload["schema"] != "mpa.diagnostic.minimax_h3_attention_capture.v1":
            raise AssertionError("unexpected Dense capture schema")
        cls.layer = int(payload["layer"])
        cls.prefixed_qkv = tuple(payload[name].cuda() for name in ("q", "k", "v"))
        cls.dense_output = payload["dense_output"].cuda()
        cls.visual_qkv = tuple(
            tensor[:, _PREFIX_TOKENS:].to(
                memory_format=torch.contiguous_format,
                copy=True,
            )
            for tensor in cls.prefixed_qkv
        )

    def _assert_output_contract(self, output: torch.Tensor, query: torch.Tensor) -> None:
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(output.dtype, query.dtype)
        self.assertTrue(torch.isfinite(output).all().item())

    @staticmethod
    def _dense_relative_nl2(output: torch.Tensor, dense: torch.Tensor) -> float:
        difference = output.float()
        difference.sub_(dense)
        error = torch.linalg.vector_norm(difference)
        return float((error / torch.linalg.vector_norm(dense.float())).item())

    @torch.inference_mode()
    def test_public_stable_precision_matrix(self) -> None:
        layouts = (
            (
                "visual_only",
                self.visual_qkv,
                anemoi.VisualLayout(_VIDEO_SHAPE),
            ),
            (
                "prefixed",
                self.prefixed_qkv,
                anemoi.VisualLayout(_VIDEO_SHAPE, prefix_tokens=_PREFIX_TOKENS),
            ),
        )
        for query_block_size in (64, 128):
            for layout_name, (query, key, value), layout in layouts:
                for precision_name, ratios in _PRECISION_CASES:
                    prefix_kv, prefix_query = _prefix_precisions(ratios)
                    with self.subTest(
                        query_block_size=query_block_size,
                        layout=layout_name,
                        precision=precision_name,
                    ):
                        output = anemoi.anemoi_attention(
                            query,
                            key,
                            value,
                            layout=layout,
                            layer=self.layer,
                            sparse_config=anemoi.SparseConfig(query_block_size=query_block_size),
                            quant_config=anemoi.QuantConfig(
                                nvfp4_ratio=ratios[0],
                                int8_ratio=ratios[1],
                                fp16_ratio=ratios[2],
                                prefix_kv_precision=prefix_kv,
                                prefix_query_precision=prefix_query,
                            ),
                        )
                        self._assert_output_contract(output, query)

    @torch.inference_mode()
    def test_generic_and_h3_nvfp4_calibration_paths(self) -> None:
        query, key, value = self.prefixed_qkv
        layout = anemoi.VisualLayout(_VIDEO_SHAPE, prefix_tokens=_PREFIX_TOKENS)
        quant = anemoi.QuantConfig(
            nvfp4_ratio=1.0,
            int8_ratio=0.0,
            prefix_kv_precision="nvfp4",
            prefix_query_precision="fp16",
        )
        unity = anemoi.anemoi_attention(
            query,
            key,
            value,
            layout=layout,
            layer=self.layer,
            sparse_config=anemoi.SparseConfig(query_block_size=128),
            quant_config=quant,
        )
        self._assert_output_contract(unity, query)

        explicit = anemoi.anemoi_attention(
            query,
            key,
            value,
            layout=layout,
            layer=self.layer,
            sparse_config=anemoi.SparseConfig(query_block_size=128),
            quant_config=quant,
            calibration=anemoi.NVFP4Calibration(0.01, 0.02, 0.03),
        )
        self._assert_output_contract(explicit, query)

        h3 = H3MPAAttention(
            H3MPAConfig(
                video_shape=_VIDEO_SHAPE,
                prefix_tokens=_PREFIX_TOKENS,
                query_block_size=128,
                nvfp4_ratio=1.0,
                int8_ratio=0.0,
                prefix_kv_precision="nvfp4",
                prefix_query_precision="fp16",
            )
        )
        calibrated = h3.mpa(
            query.squeeze(0),
            key.squeeze(0),
            value.squeeze(0),
            layer=self.layer,
        )
        self._assert_output_contract(calibrated, query.squeeze(0))

        unity_nl2 = self._dense_relative_nl2(unity, self.dense_output)
        explicit_nl2 = self._dense_relative_nl2(explicit, self.dense_output)
        h3_nl2 = self._dense_relative_nl2(calibrated.unsqueeze(0), self.dense_output)
        self.assertNotEqual(explicit_nl2, unity_nl2)
        self.assertNotEqual(h3_nl2, unity_nl2)


if __name__ == "__main__":
    unittest.main()
