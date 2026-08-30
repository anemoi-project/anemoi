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
ptxas info    : Compiling entry function 'mixed_attention_sm120_q64_int8_fp16_kernel_ILj128ELb1ELb1ELb0EE' for 'sm_120a'
ptxas info    : Function properties for mixed_attention_sm120_q64_int8_fp16_kernel_ILj128ELb1ELb1ELb0EE
    24 bytes stack frame, 36 bytes spill stores, 32 bytes spill loads
ptxas info    : Used 168 registers, used 1 barriers
"""


class SM120Q64ResourceTests(unittest.TestCase):
    def test_sass_gate_requires_fresh_q64_and_q128_ids_before_fp16_q(self) -> None:
        from scripts.check_sm120_q64_resources import validate_phase_boundary_sass

        fixture = """
//--------------------- .text.mixed_attention_sm120_q64_int8_fp16_kernel_ILj128ELb1ELb1ELb0EE --------------------------
        /*0010*/ S2R R1, SR_CTAID.Z ;
        /*0020*/ S2R R2, SR_CTAID.Y ;
        /*8000*/ S2R R3, SR_CTAID.Z ;
        /*8010*/ S2R R4, SR_CTAID.Y ;
//## File "q64_attention_phase_composer.inl", line 960
        /*8020*/ IMAD R5, R3, R6, R4 ;
//--------------------- .text.mixed_attention_sm120_q128_int8_kernel_ILj128ELb1ELb1ELb0EE --------------------------
        /*0010*/ S2UR UR1, SR_CTAID.Z ;
        /*0020*/ S2UR UR2, SR_CTAID.Y ;
        /*f000*/ S2R R3, SR_CTAID.Z ;
        /*f010*/ S2R R4, SR_CTAID.Y ;
//## File "q64_attention_phase_composer.inl", line 960
        /*f020*/ IMAD R5, R3, R6, R4 ;
"""
        result = validate_phase_boundary_sass(fixture, q_head_row_line=960)
        self.assertEqual(set(result), {"q64_int8_fp16", "q128_int8_fp16"})

    def test_sass_gate_rejects_an_entry_id_reused_by_fp16_q(self) -> None:
        from scripts.check_sm120_q64_resources import validate_phase_boundary_sass

        fixture = """
//--------------------- .text.mixed_attention_sm120_q64_int8_fp16_kernel_ILj128ELb1ELb1ELb0EE --------------------------
        /*0010*/ S2UR UR7, SR_CTAID.Z ;
        /*0020*/ S2UR UR8, SR_CTAID.Y ;
//## File "q64_attention_phase_composer.inl", line 960
        /*8000*/ UIMAD UR4, UR7, UR4, UR8 ;
"""
        with self.assertRaisesRegex(ValueError, "fresh CTAID.Z"):
            validate_phase_boundary_sass(fixture, q_head_row_line=960)

    def test_phase_resource_collector_classifies_complete_native_matrix(self) -> None:
        from scripts.check_sm120_q64_resources import collect_phase_resources

        symbols = {
            "q64_fp16": "mixed_attention_sm120_q64_kernel_ILj128ELb0ELb1ELb0EE",
            "q64_int8": "mixed_attention_sm120_q64_int8_kernel_ILj128ELb1ELb0ELb0EE",
            "q64_int8_fp16": "mixed_attention_sm120_q64_int8_fp16_kernel_ILj128ELb1ELb1ELb0EE",
            "q64_mxfp8": "mixed_attention_sm120_q64_kernel_ILj128ELb1ELb0ELb0EE",
            "q64_mxfp8_fp16": "mixed_attention_sm120_q64_kernel_ILj128ELb1ELb1ELb0EE",
            "q64_nvfp4": "mixed_attention_sm120_q64_nvfp4_kernel_ILj128ELb1ELb0ELb0EE",
            "q64_nvfp4_fp16": "mixed_attention_sm120_q64_nvfp4_kernel_ILj128ELb1ELb1ELb0EE",
            "q64_nvfp4_int8": "mixed_attention_sm120_q64_nv_int8_fp16_kernel_ILj128ELb1ELb0ELb0EE",
            "q64_nvfp4_int8_fp16": "mixed_attention_sm120_q64_nv_int8_fp16_kernel_ILj128ELb1ELb1ELb0EE",
            "q64_nvfp4_mxfp8": "mixed_attention_sm120_q64_nv_mx_fp16_kernel_ILj128ELb1ELb0ELb0EE",
            "q64_nvfp4_mxfp8_fp16": "mixed_attention_sm120_q64_nv_mx_fp16_kernel_ILj128ELb1ELb1ELb0EE",
            "q128_fp16": "mixed_attention_sm120_q128_fp16_kernel_ILj128ELb0ELb1ELb0EE",
            "q128_int8": "mixed_attention_sm120_q128_int8_kernel_ILj128ELb1ELb0ELb0EE",
            "q128_int8_fp16": "mixed_attention_sm120_q128_int8_kernel_ILj128ELb1ELb1ELb0EE",
            "q128_mxfp8": "mixed_attention_sm120_q128_mxfp8_kernel_ILj128ELb1ELb0ELb0EE",
            "q128_mxfp8_fp16": "mixed_attention_sm120_q128_mxfp8_kernel_ILj128ELb1ELb1ELb0EE",
            "q128_nvfp4": "mixed_attention_sm120_q128_nvfp4_kernel_ILj128ELb1ELb0ELb0EE",
            "q128_nvfp4_fp16": "mixed_attention_sm120_q128_nvfp4_kernel_ILj128ELb1ELb1ELb0EE",
            "q128_nvfp4_int8": "mixed_attention_sm120_q128_nv_int8_fp16_kernel_ILj128ELb1ELb0ELb0EE",
            "q128_nvfp4_int8_fp16": "mixed_attention_sm120_q128_nv_int8_fp16_kernel_ILj128ELb1ELb1ELb0EE",
            "q128_nvfp4_mxfp8": "mixed_attention_sm120_q128_nv_mx_fp16_kernel_ILj128ELb1ELb0ELb0EE",
            "q128_nvfp4_mxfp8_fp16": "mixed_attention_sm120_q128_nv_mx_fp16_kernel_ILj128ELb1ELb1ELb0EE",
        }
        fixture = "\n".join(
            f"""ptxas info    : Compiling entry function '{symbol}' for 'sm_120a'
ptxas info    : Function properties for {symbol}
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 168 registers, used 1 barriers"""
            for symbol in symbols.values()
        )

        self.assertEqual(set(collect_phase_resources(fixture)), set(symbols))

    def test_ptxas_gate_ignores_other_q64_phase_families(self) -> None:
        from scripts.check_sm120_q64_resources import validate_resources

        sibling = """
ptxas info    : Compiling entry function 'mixed_attention_sm120_q64_nvfp4_kernel_ILj128ELb1ELb1ELb0EE' for 'sm_120a'
ptxas info    : Function properties for mixed_attention_sm120_q64_nvfp4_kernel_ILj128ELb1ELb1ELb0EE
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 168 registers, used 1 barriers
"""
        result = validate_resources(GOOD_RECORD + sibling)
        self.assertEqual(set(result), {"fp16", "mxfp8", "int8_fp16"})

    def test_ptxas_gate_accepts_the_donor_resource_class(self) -> None:
        from scripts.check_sm120_q64_resources import validate_resources

        result = validate_resources(GOOD_RECORD)
        self.assertEqual(set(result), {"fp16", "mxfp8", "int8_fp16"})
        for name in ("fp16", "mxfp8"):
            record = result[name]
            self.assertEqual(record["arch"], "sm_120a")
            self.assertEqual(record["registers"], 168)
            self.assertEqual(record["spill_store_bytes"], 0)
            self.assertEqual(record["spill_load_bytes"], 0)
        self.assertEqual(result["int8_fp16"]["registers"], 168)
        self.assertEqual(result["int8_fp16"]["stack_frame_bytes"], 24)
        self.assertEqual(result["int8_fp16"]["spill_store_bytes"], 36)
        self.assertEqual(result["int8_fp16"]["spill_load_bytes"], 32)

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
