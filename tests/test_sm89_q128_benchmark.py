import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/benchmark_sm89_q128.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_sm89_q128", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load SM89 Q128 benchmark")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


class SM89Q128BenchmarkTests(unittest.TestCase):
    @staticmethod
    def _fit(slope: float, band: float | None = None) -> dict[str, float | None]:
        return {
            "slope_ms_per_stage": slope,
            "paired_slope_band_ms_per_stage": band,
        }

    def test_pure_phase_gate_requires_every_phase_to_pass(self) -> None:
        fits = {
            "q64_fp16_floor": self._fit(1.0),
            "q128_fp16_floor": self._fit(1.06),
            "q64_int8_floor": self._fit(0.5),
            "q128_int8_floor": self._fit(0.4),
        }
        accepted = benchmark._evaluate_pure_phase_gate(
            fits, suffix="_floor", max_q128_over_q64_slope=1.07
        )
        rejected = benchmark._evaluate_pure_phase_gate(
            fits, suffix="_floor", max_q128_over_q64_slope=1.05
        )
        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["phases"]["fp16"]["passed"])
        self.assertTrue(rejected["phases"]["int8"]["passed"])

    def test_pure_phase_gate_rejects_missing_measurements(self) -> None:
        result = benchmark._evaluate_pure_phase_gate(
            {}, suffix="_floor", max_q128_over_q64_slope=1.07
        )
        self.assertFalse(result["evaluated"])
        self.assertFalse(result["passed"])
        self.assertIn("missing pure phase fits", result["reason"])

    def test_transfer_gate_is_blocked_until_pure_phase_passes(self) -> None:
        fits = {
            "q64_mixed": self._fit(0.61, 0.02),
            "q64_int8_floor": self._fit(0.4),
            "q64_fp16_floor": self._fit(1.0),
            "q128_mixed": self._fit(0.615, 0.02),
            "q128_int8_floor": self._fit(0.4),
            "q128_fp16_floor": self._fit(1.0),
        }
        blocked = benchmark._evaluate_phase_transfer(
            fits, pure_phase_passed=False
        )
        accepted = benchmark._evaluate_phase_transfer(
            fits, pure_phase_passed=True
        )
        self.assertFalse(blocked["eligible"])
        self.assertIsNone(blocked["results"]["q128"]["passed"])
        self.assertTrue(accepted["eligible"])
        self.assertTrue(accepted["passed"])

    def test_transfer_gate_requires_paired_repeatability_band(self) -> None:
        fits = {
            "q64_mixed": self._fit(0.6),
            "q64_int8_floor": self._fit(0.4),
            "q64_fp16_floor": self._fit(1.0),
            "q128_mixed": self._fit(0.6),
            "q128_int8_floor": self._fit(0.4),
            "q128_fp16_floor": self._fit(1.0),
        }
        result = benchmark._evaluate_phase_transfer(
            fits, pure_phase_passed=True
        )
        self.assertFalse(result["eligible"])
        self.assertIsNone(result["passed"])
        self.assertFalse(result["results"]["q128"]["eligible"])
        self.assertEqual(
            result["results"]["q128"]["blocked_reason"],
            "paired provider order is required",
        )

    def test_transfer_gate_accepts_faster_than_pure_phase_model(self) -> None:
        fits = {
            "q64_mixed": self._fit(0.3, 0.01),
            "q64_int8_floor": self._fit(0.4),
            "q64_fp16_floor": self._fit(1.0),
            "q128_mixed": self._fit(0.3, 0.01),
            "q128_int8_floor": self._fit(0.4),
            "q128_fp16_floor": self._fit(1.0),
        }
        result = benchmark._evaluate_phase_transfer(
            fits, pure_phase_passed=True
        )
        self.assertTrue(result["passed"])
        self.assertTrue(
            result["results"]["q128"]["one_sided_no_regression_gate"]
        )

    def test_single_phase_specialization_control_compares_to_pure_floor(self) -> None:
        fits = {}
        for query_block in (64, 128):
            for phase, pure_slope in (("fp16", 1.0), ("int8", 0.5)):
                fits[f"q{query_block}_{phase}_floor"] = self._fit(
                    pure_slope, 0.01
                )
                fits[f"q{query_block}_composed_{phase}_only"] = self._fit(
                    pure_slope + (0.01 if query_block == 64 else 0.03), 0.01
                )
        result = benchmark._evaluate_single_phase_specialization_control(
            fits, pure_phase_passed=True
        )
        self.assertFalse(result["passed"])
        self.assertTrue(result["results"]["q64_fp16"]["passed"])
        self.assertFalse(result["results"]["q128_fp16"]["passed"])

    def test_single_phase_specialization_control_is_blocked_by_pure_floor(self) -> None:
        fits = {}
        for query_block in (64, 128):
            for phase in ("fp16", "int8"):
                fits[f"q{query_block}_{phase}_floor"] = self._fit(1.0, 0.01)
                fits[f"q{query_block}_composed_{phase}_only"] = self._fit(
                    1.0, 0.01
                )
        result = benchmark._evaluate_single_phase_specialization_control(
            fits, pure_phase_passed=False
        )
        self.assertFalse(result["eligible"])
        self.assertIsNone(result["passed"])
        self.assertEqual(
            result["results"]["q128_fp16"]["blocked_reason"],
            "pure phase gate failed",
        )

    def test_single_phase_specialization_control_accepts_faster_codegen(self) -> None:
        fits = {}
        for query_block in (64, 128):
            for phase in ("fp16", "int8"):
                fits[f"q{query_block}_{phase}_floor"] = self._fit(1.0, 0.01)
                fits[f"q{query_block}_composed_{phase}_only"] = self._fit(
                    0.9, 0.01
                )
        result = benchmark._evaluate_single_phase_specialization_control(
            fits, pure_phase_passed=True
        )
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
