/* DraftMap probability and fixed-budget route registration. */

#include <pybind11/pybind11.h>
#include <torch/library.h>

#include "../api.h"

TORCH_LIBRARY_FRAGMENT(mixed_attention, m) {
  m.def("draft(Tensor q_pool, Tensor k_pool) -> Tensor");
  m.def("draft_logits(Tensor q_pool, Tensor k_pool) -> Tensor");
  m.def(
      "route(Tensor probability_fp16, int n16, int n8, int n4) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "route_spatial_cross(Tensor probability_fp16, int n16, int n8, "
      "int n4, int frames, int patches_h, int patches_w) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mixed_attention, CUDA, m) {
  m.impl("draft", &draft_probability);
  m.impl("draft_logits", &draft_logits);
  m.impl("route", &route_precision);
  m.impl("route_spatial_cross", &route_precision_spatial_cross);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  (void)module;
}
