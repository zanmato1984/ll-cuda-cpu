extern "C" __device__ void loop_cuda(float *in, float *out, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    out[i] = sqrt(in[i]);
  }
}
