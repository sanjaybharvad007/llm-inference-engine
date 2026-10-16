// Host-side stand-ins for CUDA constructs so the kernel header can be
// syntax/type-checked with a plain C++ compiler in CI (no GPU, no nvcc):
//   g++ -fsyntax-only -std=c++17 -x c++ -include csrc/cuda_host_stub.h csrc/syntax_check.cpp
// Nothing here is meant to run.
#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __shared__ static
#define __launch_bounds__(...)
#define MINIVLLM_HOST_STUB 1

struct mv_dim3 { unsigned x = 1, y = 1, z = 1; };
static mv_dim3 threadIdx, blockIdx, blockDim, gridDim;
inline void __syncthreads() {}
inline float __shfl_xor_sync(unsigned, float v, int) { return v; }
inline float __expf(float x) { return expf(x); }
using std::min;

struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct uint4 { unsigned x, y, z, w; };
struct __half { unsigned short v; };
struct __half2 { __half x, y; };
struct __nv_bfloat16 { unsigned short v; };
struct __nv_bfloat162 { __nv_bfloat16 x, y; };
inline float __half2float(__half) { return 0.f; }
inline __half __float2half(float) { return __half{}; }
inline float2 __half22float2(__half2) { return float2{}; }
inline float __bfloat162float(__nv_bfloat16) { return 0.f; }
inline __nv_bfloat16 __float2bfloat16(float) { return __nv_bfloat16{}; }
inline float2 __bfloat1622float2(__nv_bfloat162) { return float2{}; }
