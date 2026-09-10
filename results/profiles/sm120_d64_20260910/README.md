# SM120 D64 implementation and 5090 results

实现已提交到 `feature/sm120-d64-20260910`，供人工审查。D64 沿用共享模板，覆盖现有 D128 的 native phase/preparation 能力；没有新增独立 D64 内核文件或 fallback。建议进入 master 合入审查，由用户检查后决定合入。

## Revision and scope

- 接手时 HEAD：`913902dbca9797a65453cc887778444a68859e4d`。
- 同步后的功能基线：`4e85afba741bdeaf2d9486cab19cb76d3e7985a4`；本地 `master`、`origin/master`、私有 `evg_tmp/master` 一致。私有 master 从其直接父提交 fast-forward 到该版本。
- 最终实现 HEAD：`d511e67cd62003fd2a530aa350cec149ce818898`；后续提交仅携带本报告和测量 JSON。完整分支 HEAD 由交付消息给出。
- 原本地 master `3846d43` 保存在 `backup/master-before-sm120-d64-20260910`；原工作分支和已有未跟踪文件均保留。
- 本次没有合入 master。没有采用需要单独维护的性能实现。

核心修改：

- [q64_attention.cuh](../../../csrc/attention/cuda/sm120/q64_attention.cuh)：D64 MXFP8 每行仅加载 2 个 scale 字节；NVFP4 使用真实 D64 fragment 和 `ldmatrix.x2`；PV 按实际 D 展开。
- [phase composer](../../../csrc/attention/cuda/sm120/q64_attention_phase_composer.inl)：D64 INT8 Q swizzle，保留现有 phase 顺序和 D128 调度。
- [host dispatch](../../../csrc/attention/cuda/sm120/q64_attention_host.cu) 与 [现有实例文件](../../../csrc/attention/cuda/sm120/instantiations)：统一运行时 D 分发，25 个 D64 与 25 个 D128 specialization；继续使用原有 16 个 TU。
- [preparation](../../../csrc/attention/cuda/sm120/q128_microscaling_preparation.cu)：standalone/fused Q/K/V、Mean/MaxPool、prefix INT8、BF16 narrowing 的布局泛化；Q128/D64 保留 128-thread metadata staging 和全部 barrier，INT8 只归约两个有效 channel warp。D128 通道判断在编译期消除。
- [K-tail](../../../csrc/attention/cuda/sm120/h3_draft_probability.cu)、[Q64 wrapper](../../../anemoi/layers/attention/mpa/backends/sm120_q64.py)、[Q128 wrapper](../../../anemoi/layers/attention/mpa/backends/sm120_q128.py)、[executor](../../../anemoi/layers/attention/mpa/executor.py)：descriptor 宽度、prefix softmax scale 和公共输入检查按实际 D 处理。

## Capability matrix

| 能力 | Q64 / D64 | Q128 / D64 |
|---|---|---|
| FP16、INT8、MXFP8、NVFP4 pure | 通过 | 通过 |
| INT8 / MXFP8 / NVFP4 + FP16 | 通过 | 通过 |
| NVFP4 + INT8，含可选 FP16 | 通过 | 通过 |
| NVFP4 + MXFP8，含可选 FP16 | 通过 | 通过 |
| prefix KV FP16 / INT8 / NVFP4；prefix query FP16 / INT8 | 通过 | 通过 |
| native GQA，Hq=4 / Hkv=2 | 通过 | 通过 |
| ragged K64 最后 5 token、非对齐 prefix、empty route / positive zero | 通过 | 通过 |
| standalone MXFP8 / NVFP4 与 fused H3 preparation | 通过 | 通过 |
| MeanPool、MaxPool、合法 anchors、FP16/BF16 public input | 通过 | 通过 |
| public `anemoi_attention` | 通过 | 通过 |
| K-tail r1 / r2（D128 当前可达路径） | 通过 | 原有 Q64 限制保持 |
| compact MXFP8 compute ceiling | 原有 Q128 限制保持 | 通过 |

GQA 是 native attention operator 的能力；公共 H3/public API 仍遵守 D128 已有的相同 Q/K/V head-shape 合约。MXFP8 沿用已有内部接口，公共 QuantConfig 的精度集合不变。纯 FP16 packing/assembly 和 FP16 prefix query 继续使用现有公共路径；没有增加 SM89 extension 依赖。

## Inputs and correctness

输入来自已索引的真实 Cosmos3 Dense inference capture：
`/home/zju/work/zxl/cosmos3_evg_benchmark/results/precision_5090/cosmos3_dense_fp16_qkv.pt`。
原张量 D128；本次显式截取前 D64/D128 channels、4 Q heads / 2 KV heads 作为 operator workload，每个 workload 重新计算 Dense。没有随机 Q/K/V，也没有使用 sparse-to-sparse error。**这不是原生 D64 模型的端到端质量评估。**

[CUDA tests](../../../tests/test_sm120_d64_cuda.py) 包括独立 Q/K scale/packing 检查、standalone/fused Q/K/V 一致性、padding、BF16 narrowing、GQA、完整与 ragged K64、11 phase families、prefix、Mean/MaxPool、anchors、K-tail、empty assembly 和 allocator churn。FP16 对 FP32 Dense 使用 `atol=2e-3, rtol=2e-2`；INT8/MXFP8 rel-L2 < 0.08、cosine > 0.995；含 NVFP4 rel-L2 < 0.30、cosine > 0.95。阈值没有因失败而放宽。

环境：RTX 5090 / SM120；CUDA toolkit 13.1；Torch 2.11.0+cu130；Python `/home/zju/miniconda3/envs/evg/bin/python`。执行前清除环境中继承的 `LD_LIBRARY_PATH` / `PYTHONPATH`。

```bash
cd /home/zju/work/evg
unset LD_LIBRARY_PATH PYTHONPATH
export ANEMOI_SM120_D64_CAPTURE=/home/zju/work/zxl/cosmos3_evg_benchmark/results/precision_5090/cosmos3_dense_fp16_qkv.pt
PY=/home/zju/miniconda3/envs/evg/bin/python
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest discover -s tests
ANEMOI_SM120_TEST_D=64 "$PY" -m unittest tests.test_sm120_d64_cuda -v
ANEMOI_SM120_TEST_D=128 "$PY" -m unittest tests.test_sm120_d64_cuda -v
```

结果：完整主机 suite 275 tests，33 项按 CUDA/外部资源条件 skip，其余通过；D64 与 D128 各 7 tests 通过。未改动 master 的隔离基线通过同一组检查（当时 raw-Q-tail 检查位于 phase test 内，共 6 methods）。

构建：`MPA_BUILD_COMPONENTS=sm120` 生成原始基线的公共 CUDA + SM120 extension；候选使用 `MPA_BUILD_COMPONENTS=sm120_q64`，复用公共 CUDA extension。

```bash
MPA_PYTHON="$PY" MPA_CUDA_HOME=/usr/local/cuda-13.1 \
MPA_BUILD_COMPONENTS=sm120_q64 MPA_MAX_JOBS=4 \
MPA_BUILD_ROOT=/home/zju/work/resources/sm120_d64_20260910/candidate/build \
scripts/build_attention_cuda.sh
```

## Sanitizer results and limits

| 检查 | 范围 | 结果 |
|---|---|---|
| memcheck | D64 全部 7 tests | 0 errors |
| synccheck | D64 全部 7 tests | 0 errors |
| initcheck global | D64 全部 7 tests | 0 errors |
| initcheck shared | 独立 preparation test，涵盖 Q64/Q128、FP16/BF16、全部量化和 prefix | 0 errors |

```bash
SAN=/usr/local/cuda-13.1/bin/compute-sanitizer
"$SAN" --tool memcheck --error-exitcode 86 "$PY" -m unittest tests.test_sm120_d64_cuda -v
"$SAN" --tool synccheck --error-exitcode 86 "$PY" -m unittest tests.test_sm120_d64_cuda -v
"$SAN" --tool initcheck --initcheck-address-space global --error-exitcode 86 \
  "$PY" -m unittest tests.test_sm120_d64_cuda -v
"$SAN" --tool initcheck --initcheck-address-space shared --kernel-name kns=prepare_ \
  --error-exitcode 86 "$PY" -m unittest \
  tests.test_sm120_d64_cuda.SM120D64CudaTests.test_preparation_layout_scales_padding_and_pooling -v
```

不宣称全应用 shared-initcheck 无条件通过：全应用首次报告来自 Dense 参考的 cuBLAS SIMT SGEMM；原始未 padding 的 Q64 FP16 直连 operator 会读取不输出的 Q-tail shared rows，未修改的 D128 基线也复现该告警，public producer 始终 padding。另一次 filtered shared-initcheck 在 Q128 INT8→FP16 返回 `cudaErrorInvalidAddressSpace`；同输入普通执行、memcheck、synccheck、global-initcheck 均通过。本次将这两项保留为非阻塞诊断，未为非 public 的 masked Q rows 改写公共 mainloop；完整 mixed-attention shared-initcheck 仍有检查范围限制。

日志保存在 `/home/zju/work/resources/sm120_d64_20260910/{baseline,candidate}/`；最终日志为 `d64_final.log`、`d128_final.log`、`host_regression_all-final.log`、`memcheck-final-seven.log`、`synccheck-final-seven.log`、`initcheck-global-final-seven.log`、`initcheck-preparation-only.log`。

## D64 performance and resources

[Benchmark script](../../../scripts/benchmark_sm120_d64.py)，CUDA Graph 21 rounds × 32 calls，median。所有 sparse phases 使用相同的 **75% effective sparsity**，固定连续 block routes；仅计 attention core，preparation 单独计时，不含路由与 assembly。Dense 使用同维度、同 Q/K/V 的 FlashAttention SDPA。

这些固定 routes 用于 kernel 吞吐诊断，不能作为真实 routing 策略的质量或端到端加速结论。其 D64 Dense-relative rel-L2 范围 0.966–2.355，已完整保留在 JSON；正确性判断使用上面的 all-kept Dense tests。

| D64 / N | Dense Q64 context (μs) | Dense Q128 context (μs) |
|---|---:|---:|
| 2048 | 33.969 | 34.020 |
| 8192 | 422.187 | 421.864 |

全部 25 个 D64 attention specialization 的 static shared、local/stack、spill store/load 均为 0。下表 dynamic shared 以 KiB 计，active warps = CTA × 4。

| Q | Phase | N2048 μs | N8192 μs | Reg/thread | Dynamic shared KiB | Active CTA/SM |
|---|---|---:|---:|---:|---:|---:|
| 64 | fp16 | 12.187 | 83.795 | 128 | 16.00 | 4 |
| 64 | int8 | 5.776 | 48.386 | 136 | 12.25 | 3 |
| 64 | int8_fp16 | 9.127 | 73.115 | 167 | 16.00 | 3 |
| 64 | mxfp8 | 6.633 | 47.472 | 127 | 12.25 | 4 |
| 64 | mxfp8_fp16 | 9.921 | 64.135 | 128 | 16.00 | 4 |
| 64 | nvfp4 | 6.181 | 35.901 | 117 | 12.25 | 4 |
| 64 | nvfp4_fp16 | 9.454 | 58.560 | 128 | 16.00 | 4 |
| 64 | nvfp4_int8 | 6.570 | 38.701 | 128 | 12.25 | 4 |
| 64 | nvfp4_int8_fp16 | 10.240 | 55.726 | 128 | 16.00 | 4 |
| 64 | nvfp4_mxfp8 | 6.819 | 42.223 | 127 | 12.25 | 4 |
| 64 | nvfp4_mxfp8_fp16 | 10.031 | 56.755 | 128 | 16.00 | 4 |
| 128 | fp16 | 15.658 | 88.501 | 253 | 24.00 | 2 |
| 128 | int8 | 8.494 | 43.372 | 255 | 16.25 | 2 |
| 128 | int8_fp16 | 13.225 | 64.238 | 248 | 24.00 | 2 |
| 128 | mxfp8 | 9.600 | 46.595 | 168 | 16.25 | 3 |
| 128 | mxfp8_fp16 | 13.736 | 65.342 | 255 | 24.00 | 2 |
| 128 | nvfp4 | 8.527 | 36.609 | 181 | 16.25 | 2 |
| 128 | nvfp4_fp16 | 12.214 | 60.010 | 246 | 24.00 | 2 |
| 128 | nvfp4_int8 | 9.495 | 40.500 | 182 | 16.25 | 2 |
| 128 | nvfp4_int8_fp16 | 13.719 | 57.008 | 244 | 24.00 | 2 |
| 128 | nvfp4_mxfp8 | 9.651 | 42.218 | 183 | 16.25 | 2 |
| 128 | nvfp4_mxfp8_fp16 | 13.794 | 57.060 | 241 | 24.00 | 2 |

另外三个 specialization 已做数值/资源检查：Q64 INT8 prefix 为 128 registers / 12.25 KiB / 4 CTA；Q128 INT8 prefix 为 183 / 16.25 KiB / 2 CTA；Q128 compact MXFP8 为 194 / 16.25 KiB / 2 CTA。本次未单独测量这三个入口的吞吐。

Q128 D64 mixed phases 仍受 241–255 registers/thread 限制，只有 2 CTA/SM；Q64 多数组合达到 4 CTA/SM。D64 减少了实际 channel work 和 shared footprint；当前数据没有支持建立独立实现的充分收益证据，因此保留共享实现。

## D128 regression

所有 25 个 D128 attention 的寄存器、static/dynamic shared、stack/local、spill、active CTA 与原始基线完全一致；11 个 fused-preparation 实例的编译资源也一致。

| Q | N | Core latency change range | Core median change |
|---|---:|---:|---:|
| 64 | 2048 | -0.95% … +2.09% | +0.42% |
| 128 | 2048 | -0.38% … +0.29% | -0.09% |
| 64 | 8192 | -4.20% … +3.18% | -0.24% |
| 128 | 8192 | -1.39% … -0.09% | -0.91% |

44 个 D128 phase/workload cells 的延迟变化中位数 −0.09%，范围 −4.20%～+3.18%；24 个 preparation cells 中位数 −0.19%，范围 −4.97%～+1.93%。Dense 基线同步保留。当前样本没有明显性能回退；微秒级差异不作为优化收益声明。

原始记录：[baseline core](baseline_benchmark.json)、[baseline preparation](baseline_preparation.json)、[candidate](candidate_benchmark.json)、[preparation resources](preparation_resources.json)、[regression](regression.json)。

```bash
"$PY" scripts/benchmark_sm120_d64.py --dims 64 128 \
  --build-log /home/zju/work/resources/sm120_d64_20260910/candidate/build-preparation.log \
  --binary anemoi/layers/attention/mpa/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so \
  --output results/profiles/sm120_d64_20260910/candidate_benchmark.json
```

## Review recommendation

建议审查本分支后决定合入 master：native D64 功能、Dense 数值、有效内存访问、barrier、preparation shared initialization 和 D128 资源/性能已验证。需要保留两项边界：没有原生 D64 模型端到端质量数据，完整 attention shared-initcheck 尚有上述告警/运行限制。独立性能实现仍需额外证据和用户检查；本次没有引入。
