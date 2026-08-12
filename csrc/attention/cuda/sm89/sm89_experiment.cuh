#pragma once

#include <cuda_runtime.h>

namespace mpa::attention {

inline bool sm89_execution_device(const cudaDeviceProp* properties) {
  return properties->major == 8 && properties->minor == 9;
}

}  // namespace mpa::attention
