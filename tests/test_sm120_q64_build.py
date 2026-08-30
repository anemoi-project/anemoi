import hashlib
import os
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class SM120Q64BuildTests(unittest.TestCase):
    def test_quantized_phases_defer_fp16_count_to_the_phase_boundary(self) -> None:
        mainloop = (
            ROOT / "csrc/attention/cuda/sm120/q64_attention.cuh"
        ).read_text()
        composer = (
            ROOT / "csrc/attention/cuda/sm120/q64_attention_phase_composer.inl"
        ).read_text()
        condition = (
            "#if MPA_LOW4_NVFP4 || MPA_MIDDLE_INT8 || MPA_MIDDLE_MXFP8"
        )
        entry = mainloop.index("const uint32_t initial_high_iterations")
        boundary = composer.index("const uint32_t high_iterations")
        self.assertIn(condition, mainloop[entry - 100 : entry])
        self.assertIn(condition, composer[boundary - 100 : boundary])

    def test_q64_int8_fp16_has_an_isolated_three_cta_specialization(self) -> None:
        mainloop = (
            ROOT / "csrc/attention/cuda/sm120/q64_attention.cuh"
        ).read_text()
        pure = (
            ROOT
            / "csrc/attention/cuda/sm120/instantiations/inst_q64_k64_d128_int8.cu"
        )
        mixed = (
            ROOT
            / "csrc/attention/cuda/sm120/instantiations/inst_q64_k64_d128_int8_fp16.cu"
        )
        self.assertTrue(mixed.is_file())
        pure_source = pure.read_text()
        mixed_source = mixed.read_text()
        self.assertIn("MPA_MIN_BLOCKS_PER_SM", mainloop)
        self.assertIn("#if defined(MPA_MIN_BLOCKS_PER_SM)", mainloop)
        self.assertNotIn("MPA_MIN_BLOCKS_PER_SM", pure_source)
        self.assertIn("<128, true, false, false>", pure_source)
        self.assertNotIn("<128, true, true, false>", pure_source)
        self.assertIn("#define MPA_MIN_BLOCKS_PER_SM 3", mixed_source)
        self.assertIn("<128, true, true, false>", mixed_source)
        setup = (ROOT / "setup.py").read_text()
        host = (ROOT / "csrc/attention/cuda/sm120/q64_attention_host.cu").read_text()
        self.assertIn('inst_q64_k64_d128_int8_fp16.cu"', setup)
        self.assertIn("launch_mixed_attention_sm120_q64_int8_fp16", host)

    def test_setup_registers_cuda_sources_with_wheel_safe_relative_paths(self) -> None:
        setup = (ROOT / "setup.py").read_text()

        self.assertIn("def _source(path: Path) -> str:", setup)
        self.assertNotIn("str(sm89 /", setup)
        self.assertNotIn("str(sm120 /", setup)

    def test_launcher_preserves_configured_python_cuda_toolkit_detection(self) -> None:
        launcher = (ROOT / "scripts/run_minimax_h3.sh").read_text()

        self.assertIn('export MPA_CUDA_HOME="${MPA_CUDA_HOME:-${python_prefix}}"', launcher)
        self.assertIn('export MPA_CUDA_HOME="${MPA_CUDA_HOME:-${CONDA_PREFIX}}"', launcher)

    def test_launcher_rejects_mixed_sm89_sm120_process_sets(self) -> None:
        launcher = (ROOT / "scripts/run_minimax_h3.sh").read_text()

        self.assertIn("if len(set(capabilities)) != 1:", launcher)

    def test_sm120_native_route_is_architecture_owned_and_registered(self) -> None:
        setup = (ROOT / "setup.py").read_text()
        api = (ROOT / "csrc/attention/cuda/sm120/api.h").read_text()
        bindings = (ROOT / "csrc/attention/cuda/sm120/bindings.cpp").read_text()
        source = ROOT / "csrc/attention/cuda/sm120/h3_route_precision.cu"
        self.assertTrue(source.is_file())
        self.assertIn('_source(sm120 / "h3_route_precision.cu")', setup)
        for registered in (api, bindings):
            self.assertIn("sm120_h3_route_precision", registered)
            self.assertIn("sm120_h3_materialize_route", registered)
        route = source.read_text()
        self.assertIn("std::optional<torch::Tensor> anchors", api)
        self.assertIn("Tensor? anchors, Tensor? anchor_ids, int anchor_count", bindings)
        self.assertIn("apply_anchor_budget_kernel", route)
        self.assertIn("anchor_ids->data_ptr<int>()", route)
        self.assertIn("id_index < anchor_count", route)
        self.assertIn("sort_values.Current()", route)

    def test_sm120_fp16_draft_is_architecture_owned_and_registered(self) -> None:
        setup = (ROOT / "setup.py").read_text()
        api = (ROOT / "csrc/attention/cuda/sm120/api.h").read_text()
        bindings = (ROOT / "csrc/attention/cuda/sm120/bindings.cpp").read_text()
        source = ROOT / "csrc/attention/cuda/sm120/h3_draft_probability.cu"
        self.assertTrue(source.is_file())
        self.assertIn('_source(sm120 / "h3_draft_probability.cu")', setup)
        for registered in (api, bindings):
            self.assertIn("sm120_h3_draft_probability", registered)
            self.assertIn("sm120_h3_k_tail_r1_probability", registered)
            self.assertIn("sm120_h3_k_tail_r2_probability", registered)

        draft = source.read_text()
        self.assertIn("row_softmax_fusion_fp16_kernel", draft)
        self.assertIn("maxpool_weight == 0.0", draft)
        self.assertIn("maxpool_weight == 1.0", draft)
        self.assertIn("launch_draft_gemm(q_max_pool", draft)

    def test_donor_first_h3_preparation_keeps_k32_staging_and_cooperative_metadata(
        self,
    ) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q128_microscaling_preparation.cu").read_text()
        self.assertIn("prepare_h3_qk_microscaling_kernel", source)
        self.assertIn("prepare_h3_v_microscaling_kernel", source)
        self.assertIn("value_stage_tokens = 32", source)
        self.assertIn("value_stages = 2", source)
        self.assertIn("shared_stride = HeadDim + 1", source)
        self.assertIn("threadIdx.x < task_tokens", source)
        self.assertIn("staged_token_indices", source)
        self.assertIn("staged_slot_valid", source)
        self.assertNotIn("half tile[64][128]", source)
        self.assertIn("bool has_maxpool", source)
        self.assertIn("if constexpr (HasMaxPool)", source)
        self.assertIn("token_valid", source)
        self.assertIn("pool_max", source)

        bindings = (ROOT / "csrc/attention/cuda/sm120/bindings.cpp").read_text()
        api = (ROOT / "csrc/attention/cuda/sm120/api.h").read_text()
        for registered in (bindings, api):
            self.assertIn("prepare_h3_sm120_operands", registered)

    def test_raster_preparation_accepts_native_sm120_builds(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm89/raster_preprocess.cu").read_text()
        devices = (ROOT / "csrc/attention/cuda/common/execution_device.cuh").read_text()
        self.assertIn("sm89_or_sm120_execution_device(properties)", source)
        self.assertIn("properties->major == 12 && properties->minor == 0", devices)

    def test_h3_output_assembly_accepts_native_sm120_builds(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm89/output_assembly.cu").read_text()
        start = source.index("assemble_h3_k64_output")
        body = source[start : source.index("return output;", start)]
        self.assertIn("sm89_or_sm120_execution_device(properties)", body)
        self.assertIn("H3 K64 output assembly requires sm89 or sm120", source)

    def test_q128_phase_stacks_keep_middle_formats_separate(self) -> None:
        root = ROOT / "csrc/attention/cuda/sm120/instantiations"
        mx = (root / "inst_q128_k64_d128_nv_mx_fp16.cu").read_text()
        int8 = (root / "inst_q128_k64_d128_nv_int8_fp16.cu").read_text()
        self.assertIn("#define MPA_MIDDLE_MXFP8 1", mx)
        self.assertNotIn("#define MPA_MIDDLE_INT8 1", mx)
        self.assertIn("#define MPA_MIDDLE_INT8 1", int8)
        self.assertNotIn("#define MPA_MIDDLE_MXFP8 1", int8)
        host = (ROOT / "csrc/attention/cuda/sm120/q64_attention_host.cu").read_text()
        self.assertIn("std::array<int64_t, 3> count_dims", host)
        self.assertIn("torch::IntArrayRef count_shape(count_dims)", host)

    def test_three_phase_nvfp4_uses_its_unpadded_v_stride(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q64_attention_phase_composer.inl").read_text()
        self.assertIn("const uint32_t nv_v_stride", source)
        self.assertEqual(source.count("nv_v_stride / 2"), 2)

    def test_q128_nvfp4_localizes_only_the_project_owned_phase(self) -> None:
        calibration = ROOT / "anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json"
        package_config = (ROOT / "pyproject.toml").read_text()
        self.assertIn(
            '"anemoi.models.minimax_h3" = ["assets/*.pt", "configs/*.json"]',
            package_config,
        )
        self.assertEqual(
            hashlib.sha256(calibration.read_bytes()).hexdigest(),
            "ef5413c53e459b8210c8fe3054a5462310f050ebc8a63e5d20a397a90627d455",
        )
        instantiation = (
            ROOT / "csrc/attention/cuda/sm120/instantiations/inst_q128_k64_d128_nvfp4.cu"
        ).read_text()
        self.assertIn("#define MPA_LOW4_NVFP4 1", instantiation)
        self.assertIn('#include "../q64_attention.cuh"', instantiation)
        mma = (ROOT / "csrc/attention/cuda/sm120/primitives/mma.cuh").read_text()
        self.assertIn("mma_m16n8k64_nvfp4", mma)
        self.assertIn("m16n8k64.row.col.kind::mxf4nvf4", mma)
        self.assertIn(".f32.e2m1.e2m1.f32.ue4m3", mma)
        source = (ROOT / "csrc/attention/cuda/sm120/q64_attention.cuh").read_text()
        start = source.index("void compute_nvfp4_qk")
        end = source.index("prepare_nvfp4_probability", start)
        helper = source[start:end]
        self.assertNotIn("compute_int_qk", helper)
        preparation = (
            ROOT / "csrc/attention/cuda/sm120/q128_microscaling_preparation.cu"
        ).read_text()
        self.assertIn("prepare_q128_nvfp4", preparation)

    def test_q128_nvfp4_k_permutation_makes_score_to_p_lane_local(self) -> None:
        def permute(index: int) -> int:
            local = index % 32
            return index - local + (local // 8) * 2 + ((local % 8) // 2) * 8 + local % 2

        self.assertEqual(sorted(permute(i) for i in range(64)), list(range(64)))
        self.assertEqual([permute(permute(i)) for i in range(64)], list(range(64)))
        for lane in range(4):
            first = [
                permute(tile * 8 + lane * 2 + parity) for tile in range(4) for parity in range(2)
            ]
            second = [
                permute(tile * 8 + lane * 2 + parity) for tile in range(4, 8) for parity in range(2)
            ]
            self.assertEqual(first, list(range(lane * 8, lane * 8 + 8)))
            self.assertEqual(second, list(range(32 + lane * 8, 40 + lane * 8)))

        preparation = (
            ROOT / "csrc/attention/cuda/sm120/q128_microscaling_preparation.cu"
        ).read_text()
        start = preparation.index("prepare_h3_qk_microscaling_kernel")
        end = preparation.index("prepare_h3_v_microscaling_kernel", start)
        kernel = preparation[start:end]
        self.assertIn("if (!is_query)", kernel)
        self.assertIn("const int64_t local = natural_row & 31", kernel)
        self.assertIn("((local % 8) / 2) * 8 + local % 2", kernel)

        mainloop = (ROOT / "csrc/attention/cuda/sm120/q64_attention.cuh").read_text()
        start = mainloop.index("prepare_nvfp4_probability")
        end = mainloop.index("template <uint32_t HeadDim>", start)
        self.assertNotIn("__shfl_sync", mainloop[start:end])

    def test_h3_nvfp4_qk_preparation_quantizes_neighbor_pairs(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q128_microscaling_preparation.cu").read_text()
        start = source.index("prepare_h3_qk_microscaling_kernel")
        end = source.index("prepare_h3_v_microscaling_kernel", start)
        kernel = source[start:end]
        self.assertIn("subgroup_max<16>(fabsf(value))", kernel)
        self.assertIn("__shfl_down_sync(", kernel)
        self.assertIn("if ((nv_lane & 1) == 0)", kernel)
        self.assertIn("channel / 2] = encode_e2m1_pair", kernel)
        self.assertEqual(source.count("qk_grid, 128, 0, stream"), 1)

    def test_h3_nvfp4_v_preparation_reuses_k32_shared_tile(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q128_microscaling_preparation.cu").read_text()
        start = source.index("prepare_h3_v_microscaling_kernel")
        end = source.index("void check_h3_raw_operand", start)
        kernel = source[start:end]
        self.assertIn("value_stage_tokens = 32", kernel)
        self.assertIn("value_tile[value_stage_tokens * shared_stride]", kernel)
        self.assertEqual(kernel.count("narrowed = load_narrowed_half("), 1)

    def test_q128_nvfp4_preparation_uses_native_pair_conversion(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q128_microscaling_preparation.cu").read_text()
        self.assertIn("#include <cuda_fp4.h>", source)
        self.assertIn("__nv_cvt_float2_to_fp4x2", source)
        self.assertNotIn("e2m1_rne", source)

    def test_q128_fp16_instantiation_reuses_generic_mainloop(self) -> None:
        source = (
            ROOT / "csrc/attention/cuda/sm120/instantiations/inst_q128_k64_d128_fp16.cu"
        ).read_text()
        self.assertIn("#define MPA_CTA_Q 128", source)
        self.assertIn("#define MPA_WARP_Q 32", source)
        self.assertIn('#include "../q64_attention.cuh"', source)
        self.assertNotIn("__global__", source)

    def test_setup_declares_independent_sm120_q64_component(self) -> None:
        source = (ROOT / "setup.py").read_text()
        self.assertIn('"sm120_q64"', source)
        self.assertIn('name="anemoi.layers.attention.mpa._cuda_sm120_q64"', source)
        self.assertIn(
            'sm120 / "instantiations" / "inst_q64_k64_d128_fp16.cu"',
            source,
        )

    def test_setup_executes_component_matrix(self) -> None:
        with patch("setuptools.setup"), patch.dict(os.environ, {"MPA_SKIP_CUDA_BUILD": "1"}):
            setup = runpy.run_path(str(ROOT / "setup.py"))

        cpp_extension = ModuleType("torch.utils.cpp_extension")
        cpp_extension.BuildExtension = object()
        cpp_extension.CUDAExtension = lambda **kwargs: kwargs
        torch_utils = ModuleType("torch.utils")
        torch_utils.cpp_extension = cpp_extension
        torch = ModuleType("torch")
        torch.utils = torch_utils
        modules = {
            "torch": torch,
            "torch.utils": torch_utils,
            "torch.utils.cpp_extension": cpp_extension,
        }
        attention = "anemoi.layers.attention.mpa._cuda_attention"
        sm120_q64 = "anemoi.layers.attention.mpa._cuda_sm120_q64"
        cases = (
            ("sm89", {}, [attention], "8.9"),
            ("sm120_q64", {}, [sm120_q64], "12.0a"),
            ("sm120", {}, [attention, sm120_q64], "12.0a"),
            ("sm89,sm120_q64", {}, [attention, sm120_q64], "12.0a"),
            (
                "sm120,sm120,sm120_q64",
                {"MPA_CUDA_ARCH_LIST": "12.0"},
                [attention, sm120_q64],
                "12.0",
            ),
        )
        for components, overrides, names, arch in cases:
            environment = {
                "MPA_SKIP_CUDA_BUILD": "0",
                "MPA_BUILD_COMPONENTS": components,
                **overrides,
            }
            with (
                self.subTest(components=components, overrides=overrides),
                patch.dict(os.environ, environment, clear=True),
                patch.dict(sys.modules, modules),
            ):
                extensions, _ = setup["_extensions"]()
                self.assertEqual([extension["name"] for extension in extensions], names)
                self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], arch)

    def test_build_identity_resolves_sm120_q64(self) -> None:
        source = (ROOT / "anemoi/layers/attention/mpa/build_identity.py").read_text()
        self.assertIn(
            '"sm120_q64": "anemoi.layers.attention.mpa._cuda_sm120_q64"',
            source,
        )
        self.assertIn("scripts/build_attention_cuda.sh", source)

    def test_sm120_q64_source_is_architecture_owned(self) -> None:
        root = ROOT / "csrc/attention/cuda/sm120"
        expected = {
            "q64_attention.cuh",
            "q64_attention_host.cu",
            "q64_attention_decl.cuh",
            "instantiations/inst_q64_k64_d128_fp16.cu",
        }
        files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        self.assertTrue(expected.issubset(files))
        for path in root.rglob("*"):
            if path.suffix in {".h", ".cuh", ".cu", ".cpp"}:
                self.assertNotIn("../sm89", path.read_text())

    def test_foundation_instantiates_only_all_fp16(self) -> None:
        path = ROOT / "csrc/attention/cuda/sm120/instantiations/inst_q64_k64_d128_fp16.cu"
        self.assertTrue(path.is_file())
        source = path.read_text()
        self.assertIn("#define MPA_CTA_Q 64", source)
        self.assertIn("#define MPA_WARP_Q 16", source)
        self.assertIn("<128, false, true, false>", source)
        self.assertNotIn("<128, true,", source)

    def test_mxfp8_phase_uses_one_specialization_for_all_ratios(self) -> None:
        path = ROOT / "csrc/attention/cuda/sm120/instantiations/inst_q64_k64_d128_mxfp8.cu"
        self.assertTrue(path.is_file())
        source = path.read_text()
        self.assertIn("#define MPA_CTA_Q 64", source)
        self.assertIn("#define MPA_WARP_Q 16", source)
        self.assertIn("#define MPA_MIDDLE_MXFP8 1", source)
        self.assertIn("<128, true, false, false>", source)
        self.assertIn("<128, true, true, false>", source)

    def test_q128_mxfp8_instantiation_reuses_existing_operator(self) -> None:
        path = ROOT / "csrc/attention/cuda/sm120/instantiations/inst_q128_k64_d128_mxfp8.cu"
        self.assertTrue(path.is_file())
        source = path.read_text()
        self.assertIn("#define MPA_CTA_Q 128", source)
        self.assertIn("#define MPA_WARP_Q 32", source)
        self.assertIn("#define MPA_MIDDLE_MXFP8 1", source)
        self.assertIn('#include "../q64_attention.cuh"', source)
        self.assertIn("<128, true, false, false>", source)
        self.assertIn("<128, true, true, false>", source)

        setup = (ROOT / "setup.py").read_text()
        bindings = (ROOT / "csrc/attention/cuda/sm120/bindings.cpp").read_text()
        api = (ROOT / "csrc/attention/cuda/sm120/api.h").read_text()
        self.assertIn(
            'sm120 / "instantiations" / "inst_q128_k64_d128_mxfp8.cu"',
            setup,
        )
        self.assertIn("sm120_q128_mxfp8_attention_forward(", bindings)
        self.assertIn("sm120_q128_mxfp8_attention_forward(", api)

    def test_sm89_audit_exposes_ragged_safe_pure_fp8_pipeline(self) -> None:
        instantiation = (
            ROOT / "csrc/attention/cuda/sm89/instantiations/inst_k64_d128.cu"
        ).read_text()
        source = (
            ROOT / "csrc/attention/cuda/sm89/mixed_attention_phase_composer.inl"
        ).read_text()
        bindings = (ROOT / "csrc/attention/cuda/sm89/bindings.cpp").read_text()

        self.assertIn("<128, true, false, false>", instantiation)
        self.assertIn("k64_fp8_attention_forward", bindings)
        pure = source[
            source.index("if constexpr (HasFp8)") : source.index("if constexpr (HasFp16)")
        ]
        self.assertIn("MPA_K64_BLOCK_MODE", pure)
        self.assertIn("absolute_stage = static_cast<uint32_t>(*low_delta++)", pure)
        self.assertIn("valid_k_counts[batch_id * num_physical_stages", pure)
        self.assertIn("cp_async::wait_group<1>();", pure)

    def test_q128_int8_control_reuses_sm89_optimized_mainloop(self) -> None:
        path = ROOT / "csrc/attention/cuda/sm89/instantiations/inst_q128_k64_d128.cu"
        self.assertTrue(path.is_file())
        source = path.read_text()
        self.assertIn("#define MPA_CTA_Q 128", source)
        self.assertIn("#define MPA_WARP_Q 32", source)
        self.assertIn('#include "../mixed_attention.cuh"', source)
        self.assertIn("<128, true, true, false>", source)
        self.assertIn(
            'sm89 / "instantiations" / "inst_q128_k64_d128.cu"',
            (ROOT / "setup.py").read_text(),
        )
        bindings = (ROOT / "csrc/attention/cuda/sm89/bindings.cpp").read_text()
        self.assertIn("q128_k64_mixed_attention_forward(", bindings)

    def test_sm89_precision_phases_are_composed_inside_persistent_softmax(self) -> None:
        kernel = (ROOT / "csrc/attention/cuda/sm89/mixed_attention.cuh").read_text()
        composer = (
            ROOT / "csrc/attention/cuda/sm89/mixed_attention_phase_composer.inl"
        ).read_text()

        state = kernel.index("float ro[num_tiles_q][num_tiles_v][8]")
        composition = kernel.index('#include "mixed_attention_phase_composer.inl"')
        normalization = kernel.index("float denominator = d[fq][row]", composition)
        self.assertLess(state, composition)
        self.assertLess(composition, normalization)
        self.assertIn("if constexpr (HasFp8)", composer)
        self.assertIn("if constexpr (HasFp16)", composer)
        self.assertNotIn("__noinline__", composer)

    def test_q128_int8_has_pure_and_mixed_phase_isolation_symbols(self) -> None:
        mixed_instantiation = (
            ROOT / "csrc/attention/cuda/sm89/instantiations/inst_q128_k64_d128.cu"
        ).read_text()
        int8_instantiation = (
            ROOT
            / "csrc/attention/cuda/sm89/instantiations/inst_q128_k64_d128_int8.cu"
        ).read_text()
        self.assertIn("<128, true, true, false>", mixed_instantiation)
        self.assertNotIn("<128, true, false, false>", mixed_instantiation)
        self.assertIn("<128, true, false, false>", int8_instantiation)
        bindings = (ROOT / "csrc/attention/cuda/sm89/bindings.cpp").read_text()
        self.assertIn("q128_k64_fp8_attention_forward", bindings)

    def test_q128_fp16_has_standalone_phase_audit_symbol(self) -> None:
        instantiation = (
            ROOT
            / "csrc/attention/cuda/sm89/instantiations/inst_q128_k64_d128_fp16.cu"
        ).read_text()
        self.assertIn("<128, false, true, false>", instantiation)
        mixed_instantiation = (
            ROOT / "csrc/attention/cuda/sm89/instantiations/inst_q128_k64_d128.cu"
        ).read_text()
        self.assertNotIn("<128, false, true, false>", mixed_instantiation)
        bindings = (ROOT / "csrc/attention/cuda/sm89/bindings.cpp").read_text()
        self.assertIn("q128_k64_fp16_attention_forward", bindings)

    def test_sm89_q128_fp16_reuses_the_pure_score_pipeline(self) -> None:
        kernel = (ROOT / "csrc/attention/cuda/sm89/mixed_attention.cuh").read_text()
        composer = (
            ROOT / "csrc/attention/cuda/sm89/mixed_attention_phase_composer.inl"
        ).read_text()
        self.assertIn("float rs[num_tiles_q][num_tiles_k][8]", composer)
        self.assertNotIn("reload_d128_fq0_q", composer)
        self.assertNotIn("stash_d128_fq0_scores", kernel + composer)
        self.assertNotIn("load_d128_fq0_scores", kernel + composer)

    def test_sm89_mixed_loads_fp16_count_at_the_phase_boundary(self) -> None:
        kernel = (ROOT / "csrc/attention/cuda/sm89/mixed_attention.cuh").read_text()
        composer = (
            ROOT / "csrc/attention/cuda/sm89/mixed_attention_phase_composer.inl"
        ).read_text()
        self.assertIn("const uint32_t initial_high_iterations", kernel)
        self.assertIn("HasFp8 && kCtaQ == 128 && low_iterations != 0", kernel)
        self.assertIn("HasFp8 && kCtaQ == 128", composer)
        self.assertIn("fp16_count + metadata_row", composer)
        self.assertIn("reinterpret_cast<volatile int32_t*>", composer)

    def test_sm89_q128_mixed_rematerializes_int8_mma_offsets(self) -> None:
        composer = (
            ROOT / "csrc/attention/cuda/sm89/mixed_attention_phase_composer.inl"
        ).read_text()
        self.assertEqual(
            composer.count("if constexpr (HasFp16 && kCtaQ == 128)"), 3
        )

    def test_sm89_q128_handoffs_only_online_softmax_state(self) -> None:
        kernel = (ROOT / "csrc/attention/cuda/sm89/mixed_attention.cuh").read_text()
        composer = (
            ROOT / "csrc/attention/cuda/sm89/mixed_attention_phase_composer.inl"
        ).read_text()

        self.assertIn("handoff_online_softmax_registers", kernel)
        self.assertIn('asm volatile("mov.f32 %0, %0;"', kernel)
        handoff = composer.index("handoff_online_softmax_registers(ro, m, d);")
        fp16_phase = composer.index("// Phase: FP16 QK/PV")
        self.assertLess(handoff, fp16_phase)
        self.assertEqual(composer.count("handoff_online_softmax_registers"), 1)
        self.assertIn("HasFp8 && HasFp16 && kCtaQ == 128", composer)

    def test_sm89_q64_pure_int8_uses_the_three_cta_envelope(self) -> None:
        kernel = (ROOT / "csrc/attention/cuda/sm89/mixed_attention.cuh").read_text()

        self.assertIn("kCtaQ == 64 ? 3 : 1", kernel)
        self.assertIn("kCtaQ == 64 && HasFp8 && !HasFp16", kernel)
        self.assertIn("output_warp_id", kernel)

    def test_int8_dense_sequential_instances_reuse_the_phase_body(self) -> None:
        setup = (ROOT / "setup.py").read_text()
        root = ROOT / "csrc/attention/cuda/sm120/instantiations"
        for topology in ("q64", "q128"):
            name = f"inst_{topology}_k64_d128_int8_dense.cu"
            path = root / name
            self.assertTrue(path.is_file())
            self.assertIn(f'sm120 / "instantiations" / "{name}"', setup)
            source = path.read_text()
            self.assertIn("#define MPA_MIDDLE_INT8 1", source)
            self.assertIn("#define MPA_DENSE_SEQUENTIAL 1", source)
            self.assertIn("#define MPA_STORE_LSE 0", source)
            self.assertIn('#include "../q64_attention.cuh"', source)

        composer = (ROOT / "csrc/attention/cuda/sm120/q64_attention_phase_composer.inl").read_text()
        self.assertIn("#if MPA_DENSE_SEQUENTIAL\n        absolute_stage = 0;", composer)
        self.assertIn(
            "#if MPA_DENSE_SEQUENTIAL\n          absolute_stage = iteration + 1;",
            composer,
        )
        self.assertIn(
            "#if MPA_DENSE_SEQUENTIAL\n          absolute_stage = low_iterations - 1;",
            composer,
        )
        for name in (
            "inst_q64_k64_d128_int8.cu",
            "inst_q128_k64_d128_int8.cu",
        ):
            self.assertNotIn("MPA_DENSE_SEQUENTIAL", (root / name).read_text())

    def test_sm89_optimized_int8_loop_is_shared_by_mixed_phase(self) -> None:
        source = (
            ROOT / "csrc/attention/cuda/sm89/mixed_attention_phase_composer.inl"
        ).read_text()
        low = source[source.index("if constexpr (HasFp8)") : source.index("if constexpr (HasFp16)")]
        self.assertNotIn("if constexpr (HasFp8 && !HasFp16)", low)
        self.assertNotIn("Mixed specializations retain their separate", low)
        self.assertIn("cp_async::wait_group<1>();", low)
        self.assertIn("int32_t* low_delta = low_lut;", low)

    def test_sm120_production_exposes_ragged_safe_pure_mx_pipeline(self) -> None:
        production = ROOT / "csrc/attention/cuda/sm120/instantiations/inst_q64_k64_d128_mxfp8.cu"
        self.assertTrue(production.is_file())
        self.assertIn("<128, true, false, false>", production.read_text())
        self.assertFalse(
            (
                ROOT / "csrc/attention/cuda/sm120/instantiations/"
                "inst_q64_k64_d128_mxfp8_pure_audit.cu"
            ).exists()
        )
        source = (ROOT / "csrc/attention/cuda/sm120/q64_attention_phase_composer.inl").read_text()
        pure = source[source.index("Shared by pure and mixed MXFP8 specializations") :]
        self.assertIn("cp_async::wait_group<1>();", pure)
        self.assertIn("absolute_stage = static_cast<uint32_t>(low_lut[iteration])", pure)
        self.assertIn("valid_k_counts[batch_id * num_physical_stages", pure)
        bindings = (ROOT / "csrc/attention/cuda/sm120/bindings.cpp").read_text()
        self.assertNotIn("pure_audit", bindings)

    def test_sm120_mx_pipeline_is_shared_by_mixed_specialization(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q64_attention_phase_composer.inl").read_text()
        shared = source[
            source.index("Shared by pure and mixed MXFP8 specializations") : source.index(
                "if constexpr (HasFp16)"
            )
        ]
        self.assertIn("if constexpr (HasFp8)", shared)
        self.assertNotIn("if constexpr (HasFp8 && !HasFp16)", shared)

    def test_mxfp8_phase_uses_one_compact_route_operator(self) -> None:
        setup = (ROOT / "setup.py").read_text()
        bindings = (ROOT / "csrc/attention/cuda/sm120/bindings.cpp").read_text()
        api = (ROOT / "csrc/attention/cuda/sm120/api.h").read_text()
        self.assertIn(
            'sm120 / "instantiations" / "inst_q64_k64_d128_mxfp8.cu"',
            setup,
        )
        self.assertIn("sm120_q64_mxfp8_attention_forward(", bindings)
        self.assertIn("Tensor q_fp16, Tensor k_fp16, Tensor v_fp16, Tensor block_ids", bindings)
        self.assertIn("Tensor mxfp8_block_counts, Tensor fp16_block_counts", bindings)
        self.assertNotIn("mxfp8_block_ids", bindings)
        self.assertIn("sm120_q64_mxfp8_attention_forward(", api)

    def test_mxfp8_phase_uses_block_scaled_qk_and_k64_pv(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q64_attention.cuh").read_text() + (
            ROOT / "csrc/attention/cuda/sm120/q64_attention_phase_composer.inl"
        ).read_text()
        self.assertIn(
            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4",
            source,
        )
        self.assertIn(".block_scale.scale_vec::1X", source)
        self.assertIn("compute_mxfp8_qk<HeadDim>", source)
        self.assertIn("prepare_mxfp8_probability", source)
        self.assertIn("kMxProbabilityScaleBits = 119U", source)
        self.assertIn("absolute_stage) * (HeadDim * 2)", source)

    def test_q128_mxfp8_reuses_each_k_fragment_across_query_fragments(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q64_attention.cuh").read_text()
        start = source.index("void compute_mxfp8_qk")
        end = source.index("__device__ __forceinline__ uint32_t pack_e4m3x4", start)
        helper = source[start:end]
        self.assertIn(
            "static_assert((NumTilesQ == 1 || NumTilesQ == 2) && NumTilesK == 4)",
            helper,
        )
        self.assertIn("uint32_t q_data[NumTilesQ][4]", helper)
        k_load = helper.index("smem_k.ldmatrix_m8n8x4")
        qmma_loop = helper.index("for (uint32_t fq = 0; fq < NumTilesQ; ++fq)", k_load)
        self.assertLess(k_load, qmma_loop)
        self.assertIn("pack_sage_mxfp8_probability", source)

    def test_q128_mxfp8_consumes_both_query_fragments_in_pv(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q64_attention_phase_composer.inl").read_text()
        shared = source[
            source.index("Shared by pure and mixed MXFP8 specializations") : source.index(
                "if constexpr (HasFp16)"
            )
        ]
        self.assertIn("for (uint32_t fq = 1; fq < num_tiles_q; ++fq)", shared)
        self.assertIn("ro[fq], rs[fq], smem_v8, v_scale_tile", shared)

    def test_mxfp8_probability_reuses_sm89_lane_local_packing(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q64_attention.cuh").read_text()
        start = source.index("pack_sage_mxfp8_probability")
        end = source.index("template <uint32_t TokenChunk, uint32_t HeadDim", start)
        packing = source[start:end]
        self.assertNotIn("__shfl_sync", packing)
        for expression in (
            "scores[TileBase], scores[TileBase] + 4",
            "scores[TileBase] + 2, scores[TileBase] + 6",
            "scores[TileBase + 1], scores[TileBase + 1] + 4",
            "scores[TileBase + 1] + 2, scores[TileBase + 1] + 6",
        ):
            self.assertIn(expression, packing)

    def test_mxfp8_phase_overlaps_v_copy_with_softmax(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q64_attention_phase_composer.inl").read_text()
        low_phase = source[source.index("if constexpr (HasFp8)") :]
        overlap = low_phase.index("MXFP8 V-copy/softmax overlap")
        v_copy = low_phase.index("load_fp8_V_global_to_share<", overlap)
        softmax = low_phase.index("update_mdo<", v_copy)
        wait = low_phase.index("cp_async::wait_group<0>();", softmax)
        self.assertLess(v_copy, softmax)
        self.assertLess(softmax, wait)

    def test_mxfp8_phase_stages_k_and_v_scales_once_per_cta(self) -> None:
        source = (ROOT / "csrc/attention/cuda/sm120/q64_attention_phase_composer.inl").read_text()
        self.assertIn("k_scale_tile", source)
        self.assertIn("v_scale_tile", source)
        self.assertIn("kLowScaleSmemBytes", source)
        self.assertGreaterEqual(
            source.count("cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch>"),
            2,
        )


if __name__ == "__main__":
    unittest.main()
