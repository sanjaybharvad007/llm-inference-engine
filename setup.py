"""Builds the optional CUDA extension (csrc/) when nvcc is available; otherwise
installs the pure-Python/Triton package.  Set MINIVLLM_SKIP_CUDA=1 to force the latter."""
import os

from setuptools import setup

ext_modules, cmdclass = [], {}
try:
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

    if CUDA_HOME is not None and os.environ.get("MINIVLLM_SKIP_CUDA", "0") != "1":
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")
        ext_modules.append(CUDAExtension(
            name="minivllm._C", sources=["csrc/paged_attention.cu"], include_dirs=[os.path.abspath("csrc")],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"],
                                "nvcc": ["-O3", "--use_fast_math", "-std=c++17", "-lineinfo", "--expt-relaxed-constexpr"]},
        ))
        cmdclass["build_ext"] = BuildExtension
except ImportError:
    pass

setup(ext_modules=ext_modules, cmdclass=cmdclass)
