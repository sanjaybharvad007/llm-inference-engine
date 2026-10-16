// Paged attention in CUDA C++ (vLLM-v1 style, with flash-decoding chunking).
//
// One thread block per (query token, query head); NUM_THREADS == HEAD_DIM so
// every thread owns one output dimension.  Keys are streamed through the
// block table in chunks of CHUNK positions:
//
//   1. QK:   thread t scores keys t, t+NT, ... of the chunk (dot over HEAD_DIM,
//            vectorised 16-byte loads of the key row, q held in shared memory)
//   2. softmax: block-wide max/sum via warp shuffles; the running (m, l) of
//            earlier chunks is rescaled online exactly as in FlashAttention
//   3. PV:   every thread walks the chunk's values and accumulates its own
//            dimension -> value rows are read fully coalesced, exactly once.
//
// Chunking bounds shared memory at CHUNK floats regardless of context length,
// so the same kernel serves 128k contexts.  Multi-token queries (speculative
// verification, small prefill chunks) are handled by giving every query token
// its own block with a causal bound of ctx_len - q_len + i + 1 keys; long
// prefill chunks go to the Triton kernel, which reuses K/V tiles across queries.
//
// Layouts: q/out [T, H_q, D]; k_cache/v_cache [num_blocks, BLOCK_SIZE, H_kv, D];
//          block_tables [num_seqs, max_blocks] int32; cu_seqlens_q [num_seqs+1];
//          context_lens [num_seqs].
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

namespace minivllm {

constexpr int CHUNK = 1024;      // keys scored per pass (shared memory: CHUNK * 4 bytes)
constexpr int WARP = 32;

// -- dtype helpers --------------------------------------------------------------
template <typename T> struct Vec;
template <> struct Vec<float> {
  using pack = float4;  static constexpr int N = 4;
  __device__ static inline void unpack(const pack& p, float* f) { f[0] = p.x; f[1] = p.y; f[2] = p.z; f[3] = p.w; }
  __device__ static inline float to_float(float v) { return v; }
  __device__ static inline float from_float(float v) { return v; }
};
template <> struct Vec<__half> {
  using pack = uint4;   static constexpr int N = 8;
  __device__ static inline void unpack(const pack& p, float* f) {
    const __half2* h = reinterpret_cast<const __half2*>(&p);
#pragma unroll
    for (int i = 0; i < 4; ++i) { float2 x = __half22float2(h[i]); f[2 * i] = x.x; f[2 * i + 1] = x.y; }
  }
  __device__ static inline float to_float(__half v) { return __half2float(v); }
  __device__ static inline __half from_float(float v) { return __float2half(v); }
};
template <> struct Vec<__nv_bfloat16> {
  using pack = uint4;   static constexpr int N = 8;
  __device__ static inline void unpack(const pack& p, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&p);
#pragma unroll
    for (int i = 0; i < 4; ++i) { float2 x = __bfloat1622float2(h[i]); f[2 * i] = x.x; f[2 * i + 1] = x.y; }
  }
  __device__ static inline float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ static inline __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
};

// -- reductions ------------------------------------------------------------------
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_max(float v, float* red) {
  v = warp_max(v);
  const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP;
  if (lane == 0) red[w] = v;
  __syncthreads();
  v = lane < NUM_WARPS ? red[lane] : -INFINITY;
  v = warp_max(v);
  __syncthreads();
  return v;
}
template <int NUM_WARPS>
__device__ __forceinline__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP;
  if (lane == 0) red[w] = v;
  __syncthreads();
  v = lane < NUM_WARPS ? red[lane] : 0.f;
  v = warp_sum(v);
  __syncthreads();
  return v;
}

// -- the kernel ------------------------------------------------------------------
template <typename T, int HEAD_DIM, int BLOCK_SIZE>
__global__ void __launch_bounds__(HEAD_DIM)
paged_attention_kernel(
    T* __restrict__ out, const T* __restrict__ q,
    const T* __restrict__ k_cache, const T* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables, const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ context_lens,
    int num_seqs, int num_heads, int num_kv_heads, int max_blocks_per_seq, float scale) {
  constexpr int NT = HEAD_DIM;
  constexpr int NUM_WARPS = NT / WARP;
  using V = Vec<T>;
  constexpr int VN = V::N;
  static_assert(HEAD_DIM % VN == 0, "head_dim must be a multiple of the vector width");

  __shared__ float s_q[HEAD_DIM];
  __shared__ float s_logits[CHUNK];
  __shared__ float s_red[NUM_WARPS];

  const int t = blockIdx.x;                 // query token (flattened)
  const int h = blockIdx.y;                 // query head
  const int tid = threadIdx.x;
  const int h_kv = h / (num_heads / num_kv_heads);

  // which sequence does token t belong to?  (num_seqs is small; linear scan is fine)
  int s = 0;
  while (s + 1 < num_seqs && cu_seqlens_q[s + 1] <= t) ++s;
  const int q_start = cu_seqlens_q[s], q_len = cu_seqlens_q[s + 1] - q_start;
  const int ctx_len = context_lens[s];
  const int n_keys = ctx_len - q_len + (t - q_start) + 1;   // causal: this token sees keys [0, n_keys)
  const int32_t* bt = block_tables + (size_t)s * max_blocks_per_seq;

  s_q[tid] = V::to_float(q[((size_t)t * num_heads + h) * HEAD_DIM + tid]);
  __syncthreads();

  const size_t kv_stride_tok = (size_t)num_kv_heads * HEAD_DIM;             // elements per cache token
  const size_t kv_stride_blk = (size_t)BLOCK_SIZE * kv_stride_tok;

  float m_run = -INFINITY, l_run = 0.f, acc = 0.f;

  for (int c0 = 0; c0 < n_keys; c0 += CHUNK) {
    const int n = min(CHUNK, n_keys - c0);

    // 1. scores for this chunk
    float local_max = -INFINITY;
    for (int j = tid; j < n; j += NT) {
      const int pos = c0 + j;
      const T* krow = k_cache + (size_t)bt[pos / BLOCK_SIZE] * kv_stride_blk
                      + (size_t)(pos % BLOCK_SIZE) * kv_stride_tok + (size_t)h_kv * HEAD_DIM;
      float dot = 0.f;
#pragma unroll
      for (int d = 0; d < HEAD_DIM; d += VN) {
        typename V::pack p = *reinterpret_cast<const typename V::pack*>(krow + d);
        float kf[VN];
        V::unpack(p, kf);
#pragma unroll
        for (int i = 0; i < VN; ++i) dot += kf[i] * s_q[d + i];
      }
      dot *= scale;
      s_logits[j] = dot;
      local_max = fmaxf(local_max, dot);
    }
    __syncthreads();

    // 2. online softmax update
    const float m_chunk = block_max<NUM_WARPS>(local_max, s_red);
    const float m_new = fmaxf(m_run, m_chunk);
    const float alpha = __expf(m_run - m_new);
    float local_sum = 0.f;
    for (int j = tid; j < n; j += NT) {
      const float p = __expf(s_logits[j] - m_new);
      s_logits[j] = p;
      local_sum += p;
    }
    __syncthreads();
    const float l_chunk = block_sum<NUM_WARPS>(local_sum, s_red);
    l_run = l_run * alpha + l_chunk;
    acc *= alpha;
    m_run = m_new;

    // 3. P.V — thread tid owns output dim tid; value rows are read coalesced
    for (int j = 0; j < n; ++j) {
      const int pos = c0 + j;
      const T* vrow = v_cache + (size_t)bt[pos / BLOCK_SIZE] * kv_stride_blk
                      + (size_t)(pos % BLOCK_SIZE) * kv_stride_tok + (size_t)h_kv * HEAD_DIM;
      acc += s_logits[j] * V::to_float(vrow[tid]);
    }
    __syncthreads();
  }

  out[((size_t)t * num_heads + h) * HEAD_DIM + tid] = V::from_float(acc / l_run);
}

}  // namespace minivllm
