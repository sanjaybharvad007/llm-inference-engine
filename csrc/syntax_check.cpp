// Instantiates every (dtype, head_dim, block_size) variant of the kernel under
// the host stub so CI catches template / type errors without a GPU.
#include "paged_attention_kernel.cuh"
using namespace minivllm;
template void paged_attention_kernel<float, 64, 8>(float*, const float*, const float*, const float*, const int32_t*, const int32_t*, const int32_t*, int, int, int, int, float);
template void paged_attention_kernel<float, 128, 16>(float*, const float*, const float*, const float*, const int32_t*, const int32_t*, const int32_t*, int, int, int, int, float);
template void paged_attention_kernel<__half, 128, 16>(__half*, const __half*, const __half*, const __half*, const int32_t*, const int32_t*, const int32_t*, int, int, int, int, float);
template void paged_attention_kernel<__nv_bfloat16, 64, 32>(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const int32_t*, const int32_t*, const int32_t*, int, int, int, int, float);
