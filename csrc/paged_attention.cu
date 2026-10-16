// PyTorch glue for the CUDA paged-attention kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "paged_attention_kernel.cuh"

namespace minivllm {

template <typename T, int HEAD_DIM, int BLOCK_SIZE>
static void launch(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                   const torch::Tensor& bt, const torch::Tensor& cu, const torch::Tensor& ctx, float scale) {
  const int total_q = q.size(0), num_heads = q.size(1), num_kv_heads = k.size(2);
  dim3 grid(total_q, num_heads), block(HEAD_DIM);
  auto stream = at::cuda::getCurrentCUDAStream();
  paged_attention_kernel<T, HEAD_DIM, BLOCK_SIZE><<<grid, block, 0, stream>>>(
      reinterpret_cast<T*>(out.data_ptr()), reinterpret_cast<const T*>(q.data_ptr()),
      reinterpret_cast<const T*>(k.data_ptr()), reinterpret_cast<const T*>(v.data_ptr()),
      bt.data_ptr<int32_t>(), cu.data_ptr<int32_t>(), ctx.data_ptr<int32_t>(),
      (int)cu.size(0) - 1, num_heads, num_kv_heads, (int)bt.size(1), scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int HEAD_DIM>
static void dispatch_block(int block_size, torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k,
                           const torch::Tensor& v, const torch::Tensor& bt, const torch::Tensor& cu,
                           const torch::Tensor& ctx, float scale) {
  switch (block_size) {
    case 8:  launch<T, HEAD_DIM, 8>(out, q, k, v, bt, cu, ctx, scale); break;
    case 16: launch<T, HEAD_DIM, 16>(out, q, k, v, bt, cu, ctx, scale); break;
    case 32: launch<T, HEAD_DIM, 32>(out, q, k, v, bt, cu, ctx, scale); break;
    default: TORCH_CHECK(false, "unsupported block_size ", block_size, " (8/16/32)");
  }
}

template <typename T>
static void dispatch_dim(int head_dim, int block_size, torch::Tensor& out, const torch::Tensor& q,
                         const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& bt,
                         const torch::Tensor& cu, const torch::Tensor& ctx, float scale) {
  switch (head_dim) {
    case 64:  dispatch_block<T, 64>(block_size, out, q, k, v, bt, cu, ctx, scale); break;
    case 128: dispatch_block<T, 128>(block_size, out, q, k, v, bt, cu, ctx, scale); break;
    default: TORCH_CHECK(false, "unsupported head_dim ", head_dim, " (64/128)");
  }
}

torch::Tensor paged_attention_fwd(const torch::Tensor& q, const torch::Tensor& k_cache, const torch::Tensor& v_cache,
                                  const torch::Tensor& block_tables, const torch::Tensor& cu_seqlens_q,
                                  const torch::Tensor& context_lens, double scale) {
  TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.dim() == 3, "q must be a contiguous CUDA [T, H, D] tensor");
  TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous() && k_cache.dim() == 4, "caches must be contiguous [B, bs, H_kv, D]");
  TORCH_CHECK(block_tables.dtype() == torch::kInt32 && cu_seqlens_q.dtype() == torch::kInt32 &&
              context_lens.dtype() == torch::kInt32, "index tensors must be int32");
  TORCH_CHECK(q.size(1) % k_cache.size(2) == 0, "num_heads must be a multiple of num_kv_heads");
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto out = torch::empty_like(q);
  const int head_dim = q.size(2), block_size = k_cache.size(1);
  auto bt = block_tables.contiguous(), cu = cu_seqlens_q.contiguous(), ctx = context_lens.contiguous();
  switch (q.scalar_type()) {
    case torch::kFloat16:  dispatch_dim<__half>(head_dim, block_size, out, q, k_cache, v_cache, bt, cu, ctx, (float)scale); break;
    case torch::kBFloat16: dispatch_dim<__nv_bfloat16>(head_dim, block_size, out, q, k_cache, v_cache, bt, cu, ctx, (float)scale); break;
    case torch::kFloat32:  dispatch_dim<float>(head_dim, block_size, out, q, k_cache, v_cache, bt, cu, ctx, (float)scale); break;
    default: TORCH_CHECK(false, "unsupported dtype ", q.scalar_type());
  }
  return out;
}

}  // namespace minivllm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "minivllm CUDA kernels";
  m.def("paged_attention_fwd", &minivllm::paged_attention_fwd,
        "Paged attention forward (q, k_cache, v_cache, block_tables, cu_seqlens_q, context_lens, scale)");
}
