import unittest


def _record(
    tag: str,
    registers: int,
    *,
    stack: int = 0,
    stores: int = 0,
    loads: int = 0,
    arch: str = "sm_89",
) -> str:
    symbol = f"mixed_attention_sm89_k64_kernelILj128{tag}Ev"
    return f"""ptxas info    : Compiling entry function '{symbol}' for '{arch}'
ptxas info    : Function properties for {symbol}
    {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads
ptxas info    : Used {registers} registers, 512 bytes cmem[0]
"""


GOOD_RECORD = "".join(
    (
        _record("ELb1ELb0ELb0EE", 168),
        _record("ELb1ELb1ELb0EE", 168, stack=24, stores=40, loads=52),
        _record("ELb0ELb1ELb0EE", 168),
    )
)


class SM89Q64ResourceTests(unittest.TestCase):
    def test_gate_accepts_all_compile_time_precision_variants(self) -> None:
        from scripts.check_sm89_q64_resources import validate_resources

        result = validate_resources(GOOD_RECORD)
        self.assertEqual(result["fp16"]["registers"], 168)
        self.assertEqual(result["int8"]["registers"], 168)
        self.assertEqual(result["int8_fp16"]["registers"], 168)
        self.assertEqual(result["int8_fp16"]["spill_load_bytes"], 52)

    def test_gate_rejects_local_memory(self) -> None:
        from scripts.check_sm89_q64_resources import validate_resources

        bad = GOOD_RECORD.replace("40 bytes spill stores", "44 bytes spill stores")
        with self.assertRaisesRegex(ValueError, "spill stores 44 bytes exceed limit 40"):
            validate_resources(bad)

    def test_gate_rejects_register_regression(self) -> None:
        from scripts.check_sm89_q64_resources import validate_resources

        baseline = validate_resources(GOOD_RECORD)
        bad = GOOD_RECORD.replace("Used 168 registers", "Used 169 registers", 1)
        with self.assertRaisesRegex(ValueError, "registers 169 exceed limit 168"):
            validate_resources(bad, baseline)

    def test_gate_rejects_wrong_architecture(self) -> None:
        from scripts.check_sm89_q64_resources import validate_resources

        with self.assertRaisesRegex(ValueError, "architecture sm_90"):
            validate_resources(GOOD_RECORD.replace("sm_89", "sm_90", 1))


if __name__ == "__main__":
    unittest.main()
