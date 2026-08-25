from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "csrc/attention/cuda/sm120/q64_attention.cuh"
FRAGMENT = ROOT / "csrc/attention/cuda/sm120/q64_attention_phase_composer.inl"
DECLARATIONS = ROOT / "csrc/attention/cuda/sm120/q64_attention_decl.cuh"
HOST = ROOT / "csrc/attention/cuda/sm120/q64_attention_host.cu"
API = ROOT / "csrc/attention/cuda/sm120/api.h"
BINDINGS = ROOT / "csrc/attention/cuda/sm120/bindings.cpp"
PREPARATION = ROOT / "csrc/attention/cuda/sm120/q128_microscaling_preparation.cu"
SETUP = ROOT / "setup.py"
INSTANTIATIONS = ROOT / "csrc/attention/cuda/sm120/instantiations"
PHASES = (
    "NVFP4",
    "MXFP8",
    "INT8",
    "FP16",
)


class SM120PhaseComposerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text()
        start = cls.source.index("void MPA_ATTENTION_KERNEL_ENTRY(")
        end = cls.source.index(
            "normalize_d_inplace<num_tiles_q, num_tiles_v>(ro, d);",
            start,
        )
        cls.composer = cls.source[start:end]

    def test_composer_expands_one_canonical_phase_fragment(self) -> None:
        self.assertTrue(FRAGMENT.is_file())
        include = '#include "q64_attention_phase_composer.inl"'
        self.assertEqual(self.composer.count(include), 1)
        for helper in (
            "run_nvfp4_phase",
            "run_mxfp8_phase",
            "run_int8_phase",
            "run_fp16_phase",
        ):
            self.assertNotIn(helper, self.source)

    def test_composer_keeps_only_persistent_state_between_phases(self) -> None:
        self.assertIn("float ro[num_tiles_q][num_tiles_v][8];", self.composer)
        self.assertIn("float m[num_tiles_q][2];", self.composer)
        self.assertIn("float d[num_tiles_q][2];", self.composer)
        self.assertNotIn("float rs[", self.composer)
        self.assertNotIn("probability0", self.composer)

    def test_fragment_keeps_phase_order_and_local_temporaries(self) -> None:
        fragment = FRAGMENT.read_text()
        markers = [fragment.index(f"// Phase: {phase}") for phase in PHASES]
        self.assertEqual(markers, sorted(markers))
        for local in ("float rs[", "probability0", "k_tile", "v_tile"):
            self.assertIn(local, fragment)

    def test_int8_phase_uses_the_two_group_donor_pipeline(self) -> None:
        fragment = FRAGMENT.read_text()
        start = fragment.index("// Phase: INT8")
        end = fragment.index("// Phase: FP16", start)
        phase = fragment[start:end]
        self.assertIn("Prologue: Q is already resident", phase)
        self.assertIn("iteration + 2 < low_iterations", phase)
        self.assertGreaterEqual(phase.count("cp_async::wait_group<1>();"), 4)
        self.assertIn("compute_fp8_sv_inst_buf_fp16_accu<", phase)
        self.assertIn("cp_async::wait_group<0>();", phase)
        self.assertNotIn(
            "iteration < low_iterations; ++iteration",
            phase,
        )

    def test_standalone_int8_excludes_the_legacy_mxfp8_default(self) -> None:
        fragment = FRAGMENT.read_text()
        self.assertIn(
            "#if (!MPA_LOW4_NVFP4 && !MPA_MIDDLE_INT8) || "
            "MPA_MIDDLE_MXFP8",
            fragment,
        )
        self.assertIn(
            "#if !MPA_MIDDLE_INT8 && !MPA_LOW4_NVFP4 && "
            "!MPA_MIDDLE_MXFP8",
            self.source,
        )
        self.assertNotIn("MPA_AUDIT_PURE_MX", self.source)

    def test_q64_and_q128_int8_families_compile_active_only_variants(self) -> None:
        for query_block in (64, 128):
            path = INSTANTIATIONS / f"inst_q{query_block}_k64_d128_int8.cu"
            self.assertTrue(path.is_file())
            source = path.read_text()
            self.assertIn("#define MPA_MIDDLE_INT8 1", source)
            self.assertIn("<128, true, false, false>", source)
            self.assertIn("<128, true, true, false>", source)
            self.assertIn(path.name, SETUP.read_text())

    def test_int8_public_abi_selects_active_fp16_specialization(self) -> None:
        declarations = DECLARATIONS.read_text()
        host = HOST.read_text()
        api = API.read_text()
        bindings = BINDINGS.read_text()
        for query_block in (64, 128):
            name = f"sm120_q{query_block}_int8_attention_forward"
            launcher = f"launch_mixed_attention_sm120_q{query_block}_int8"
            self.assertIn(name, api)
            self.assertIn(name, bindings)
            self.assertIn(launcher, declarations)
            self.assertIn(f"{launcher}<128, true, false, false>", host)
            self.assertIn(f"{launcher}<128, true, true, false>", host)
        self.assertIn("bool active_fp16 = true", api)
        self.assertIn("bool active_fp16=True", bindings)
        self.assertIn("fp16_block_counts.numel() == 0", host)

    def test_fp16_body_is_shared_by_pure_and_composed_families(self) -> None:
        composer = FRAGMENT.read_text()
        self.assertEqual(composer.count("// Phase: FP16"), 1)
        self.assertIn("compute_fp16_sv_stage_tilewise", composer)
        for query_block in (64, 128):
            for family in ("fp16", "int8", "mxfp8", "nvfp4"):
                source = (
                    INSTANTIATIONS
                    / f"inst_q{query_block}_k64_d128_{family}.cu"
                ).read_text()
                self.assertIn('#include "../q64_attention.cuh"', source)
            for family in ("nv_int8_fp16", "nv_mx_fp16"):
                source = (
                    INSTANTIATIONS
                    / f"inst_q{query_block}_k64_d128_{family}.cu"
                ).read_text()
                self.assertIn('#include "../q64_attention.cuh"', source)
                self.assertIn("<128, true, true, false>", source)

    def test_nvfp4_fp16_count_is_reloaded_at_the_fp16_boundary(self) -> None:
        self.assertIn(
            "HasFp16 && route_low_iterations == 0",
            self.composer,
        )
        composer = FRAGMENT.read_text()
        fp16 = composer[composer.index("// Phase: FP16") :]
        self.assertIn("#if MPA_THREE_PHASE || MPA_LOW4_NVFP4", fp16)
        self.assertIn("reinterpret_cast<volatile int32_t*>", fp16)

    def test_nvfp4_phase_supports_one_or_two_query_fragments(self) -> None:
        fragment = FRAGMENT.read_text()
        start = fragment.index("// Phase: NVFP4")
        end = fragment.index("// Phase: MXFP8", start)
        phase = fragment[start:end]
        self.assertNotIn("static_assert(kCtaQ == 128", phase)
        self.assertIn("if constexpr (num_tiles_q == 2)", phase)
        self.assertIn("prepare_nvfp4_probability(rs[0])", phase)
        self.assertIn("prepare_nvfp4_probability(rs[1])", phase)
        source = SOURCE.read_text()
        self.assertIn("NumTilesQ == 1 || NumTilesQ == 2", source)
        self.assertNotIn("NumTilesQ == 2 && NumTilesK == 4", source)

    def test_nvfp4_k32_permutation_is_shared_and_tail_safe(self) -> None:
        def permute(local: int) -> int:
            return (local // 8) * 2 + ((local % 8) // 2) * 8 + local % 2

        self.assertEqual(sorted(permute(i) for i in range(32)), list(range(32)))
        self.assertTrue(all(permute(permute(i)) == i for i in range(32)))
        self.assertTrue(all(permute(i) // 32 == i // 32 for i in range(32)))
        self.assertEqual([permute(30), permute(31)], [30, 31])
        preparation = PREPARATION.read_text()
        self.assertEqual(preparation.count("source_row = row - local"), 1)
        self.assertIn("template <uint32_t QueryBlock>", preparation)
        self.assertIn("prepare_q64_nvfp4", preparation)
        self.assertIn("prepare_q128_nvfp4", preparation)

    def test_q64_and_q128_nvfp4_families_compile_active_only_variants(self) -> None:
        for query_block in (64, 128):
            pure = INSTANTIATIONS / f"inst_q{query_block}_k64_d128_nvfp4.cu"
            stack = (
                INSTANTIATIONS
                / f"inst_q{query_block}_k64_d128_nv_int8_fp16.cu"
            )
            for path in (pure, stack):
                self.assertTrue(path.is_file())
                source = path.read_text()
                self.assertIn("<128, true, false, false>", source)
                self.assertIn("<128, true, true, false>", source)
                self.assertIn(path.name, SETUP.read_text())

    def test_nvfp4_public_abi_selects_active_fp16_specialization(self) -> None:
        declarations = DECLARATIONS.read_text()
        host = HOST.read_text()
        api = API.read_text()
        bindings = BINDINGS.read_text()
        for query_block in (64, 128):
            pure_name = f"sm120_q{query_block}_nvfp4_attention_forward"
            stack_name = f"sm120_q{query_block}_nv_int8_fp16_attention_forward"
            for name in (pure_name, stack_name):
                self.assertIn(name, api)
                self.assertIn(name, bindings)
            for launcher in (
                f"launch_mixed_attention_sm120_q{query_block}_nvfp4",
                f"launch_mixed_attention_sm120_q{query_block}_nv_int8_fp16",
            ):
                self.assertIn(launcher, declarations)
                self.assertIn(f"{launcher}<128, true, false, false>", host)
                self.assertIn(f"{launcher}<128, true, true, false>", host)
        self.assertIn("bool active_fp16 = true", api)
        self.assertIn("bool active_fp16=True", bindings)

    def test_mxfp8_families_compile_active_only_variants(self) -> None:
        setup = SETUP.read_text()
        for query_block in (64, 128):
            for family in ("mxfp8", "nv_mx_fp16"):
                path = (
                    INSTANTIATIONS
                    / f"inst_q{query_block}_k64_d128_{family}.cu"
                )
                self.assertTrue(path.is_file())
                source = path.read_text()
                self.assertIn("<128, true, false, false>", source)
                self.assertIn("<128, true, true, false>", source)
                self.assertIn(path.name, setup)

    def test_mxfp8_public_abi_selects_active_fp16_specialization(self) -> None:
        declarations = DECLARATIONS.read_text()
        host = HOST.read_text()
        api = API.read_text()
        bindings = BINDINGS.read_text()
        setup = SETUP.read_text()
        for query_block in (64, 128):
            pure_name = f"sm120_q{query_block}_mxfp8_attention_forward"
            stack_name = f"sm120_q{query_block}_nv_mx_fp16_attention_forward"
            for name in (pure_name, stack_name):
                self.assertIn(name, api)
                self.assertIn(name, bindings)
            for launcher in (
                (
                    "launch_mixed_attention_sm120_q64"
                    if query_block == 64
                    else "launch_mixed_attention_sm120_q128_mxfp8"
                ),
                f"launch_mixed_attention_sm120_q{query_block}_nv_mx_fp16",
            ):
                self.assertIn(launcher, declarations)
                self.assertIn(f"{launcher}<128, true, false, false>", host)
                self.assertIn(f"{launcher}<128, true, true, false>", host)
        for text in (SOURCE.read_text(), host, api, bindings, setup):
            self.assertNotIn("MPA_AUDIT_PURE_MX", text)
            self.assertNotIn("mxfp8_pure_audit", text)
        self.assertFalse(
            (INSTANTIATIONS / "inst_q64_k64_d128_mxfp8_pure_audit.cu").exists()
        )
        self.assertIn("bool active_fp16 = true", api)
        self.assertIn("bool active_fp16=True", bindings)
        self.assertIn("fp16_block_counts.numel() == 0", host)


if __name__ == "__main__":
    unittest.main()
