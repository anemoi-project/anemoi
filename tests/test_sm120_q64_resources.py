from types import SimpleNamespace
import unittest
from unittest.mock import patch


GOOD_RECORD = """
ptxas info    : Compiling entry function 'mixed_attention_sm120_q64_kernel_ILj128ELb0ELb1ELb0EE' for 'sm_120a'
ptxas info    : Function properties for mixed_attention_sm120_q64_kernel_ILj128ELb0ELb1ELb0EE
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 168 registers, used 1 barriers
ptxas info    : Compiling entry function 'mixed_attention_sm120_q64_kernel_ILj128ELb1ELb1ELb0EE' for 'sm_120a'
ptxas info    : Function properties for mixed_attention_sm120_q64_kernel_ILj128ELb1ELb1ELb0EE
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 168 registers, used 1 barriers
"""


class SM120Q64ResourceTests(unittest.TestCase):
    def test_ptxas_gate_accepts_the_donor_resource_class(self) -> None:
        from scripts.check_sm120_q64_resources import validate_resources

        result = validate_resources(GOOD_RECORD)
        self.assertEqual(set(result), {"fp16", "mxfp8"})
        for record in result.values():
            self.assertEqual(record["arch"], "sm_120a")
            self.assertEqual(record["registers"], 168)
            self.assertEqual(record["spill_store_bytes"], 0)
            self.assertEqual(record["spill_load_bytes"], 0)

    def test_ptxas_gate_rejects_register_growth(self) -> None:
        from scripts.check_sm120_q64_resources import validate_resources

        with self.assertRaisesRegex(ValueError, "registers"):
            validate_resources(GOOD_RECORD.replace("168 registers", "169 registers"))

    def test_ptxas_gate_rejects_spill(self) -> None:
        from scripts.check_sm120_q64_resources import validate_resources

        with self.assertRaisesRegex(ValueError, "spill"):
            validate_resources(GOOD_RECORD.replace("0 bytes spill stores", "4 bytes spill stores"))

    def test_runtime_gate_accepts_three_resident_ctas(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q64

        module = SimpleNamespace(
            sm120_q64_fp16_kernel_metadata=lambda: (168, 0, 101376, 3)
        )
        with patch.object(
            sm120_q64, "import_native_extension", return_value=module
        ):
            metadata = sm120_q64.sm120_q64_fp16_kernel_metadata()
        self.assertEqual(metadata["registers"], 168)
        self.assertEqual(metadata["active_ctas_per_sm"], 3)

    def test_runtime_gate_rejects_two_resident_ctas(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q64

        module = SimpleNamespace(
            sm120_q64_fp16_kernel_metadata=lambda: (168, 0, 101376, 2)
        )
        with patch.object(
            sm120_q64, "import_native_extension", return_value=module
        ):
            with self.assertRaisesRegex(RuntimeError, "resource gate failed"):
                sm120_q64.sm120_q64_fp16_kernel_metadata()

    def test_runtime_gate_covers_the_unified_mxfp8_binary(self) -> None:
        from anemoi.layers.attention.mpa.backends import sm120_q64

        module = SimpleNamespace(
            sm120_q64_mxfp8_kernel_metadata=lambda: (168, 0, 101376, 3)
        )
        with patch.object(
            sm120_q64, "import_native_extension", return_value=module
        ):
            metadata = sm120_q64.sm120_q64_mxfp8_kernel_metadata()
        self.assertEqual(metadata["registers"], 168)
        self.assertEqual(metadata["active_ctas_per_sm"], 3)


if __name__ == "__main__":
    unittest.main()
