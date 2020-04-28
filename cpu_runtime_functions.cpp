#include <cmath>
#include <cstddef>

/// Actual CPU loop function.
extern "C" __attribute__((always_inline)) void loop_cpu(float *in, float *out,
                                                        size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = sqrt(in[i]);
  }
}

/// Stub for CUDA loop function, will be replaced in CUDA backend.
extern "C" __attribute__((noinline)) void loop_cuda(float *in, float *out,
                                                    size_t n) {}
