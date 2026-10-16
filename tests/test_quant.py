"""Weight-only quantization: numerics, model surgery, and engine output quality."""
import os

import pytest
import torch

from minivllm.config import EngineConfig, ModelConfig, SamplingParams
from minivllm.core.engine import LLMEngine
from minivllm.models.llama import LlamaForCausalLM, reference_forward_full
from minivllm.quant import QuantConfig, QuantLinear, quantize_model, quantize_weight
from minivllm.quant.quantize import HAS_FP8, dequantize_weight
from minivllm.tokenizer import ByteTokenizer


@pytest.mark.parametrize("method", ["int8"] + (["fp8"] if HAS_FP8 else []))
def test_quantize_roundtrip_error_is_small(method):
    torch.manual_seed(0)
    w = torch.randn(256, 512) * 0.02
    wq, scale = quantize_weight(w, method)
    assert scale.shape == (256,)
    back = dequantize_weight(wq, scale, torch.float32)
    rel = (back - w).norm() / w.norm()
    assert rel < (0.01 if method == "int8" else 0.08), rel
    # per-row: max value of every row is reproduced (nearly) exactly
    assert torch.allclose(back.abs().amax(1), w.abs().amax(1), rtol=0.07)


def test_quant_linear_matches_dequantized_linear():
    torch.manual_seed(0)
    w = torch.randn(96, 128) * 0.05
    wq, scale = quantize_weight(w, "int8")
    layer = QuantLinear(wq, scale, None, torch.float32)
    x = torch.randn(7, 128)
    expect = x @ dequantize_weight(wq, scale, torch.float32).T
    assert torch.allclose(layer(x), expect, atol=1e-5)


def test_quantize_model_replaces_projections_and_keeps_lm_head():
    torch.manual_seed(0)
    model = LlamaForCausalLM(ModelConfig.tiny(hidden=64))
    stats = quantize_model(model, QuantConfig("int8", min_features=1))
    layer = model.layers[0]
    assert isinstance(layer.self_attn.q_proj, QuantLinear) and isinstance(layer.mlp.down_proj, QuantLinear)
    assert layer.mlp.down_proj.reduce and not layer.self_attn.q_proj.reduce
    assert not isinstance(model.lm_head, QuantLinear)
    assert stats["quantized"] == 2 * 7 and stats["bytes_after"] < stats["bytes_before"] / 3
    assert model.quant_config.method == "int8"


def test_quantized_model_logits_close_and_greedy_output_matches():
    torch.manual_seed(0)
    cfg = ModelConfig.tiny(hidden=128, layers=2)
    model = LlamaForCausalLM(cfg)
    toks = list(range(5, 30))
    ref = reference_forward_full(model, toks)
    quantize_model(model, QuantConfig("int8", min_features=1))
    got = reference_forward_full(model, toks)
    assert torch.allclose(got, ref, atol=0.05, rtol=0.05), (got - ref).abs().max()
    assert (got.argmax(-1) == ref.argmax(-1)).float().mean() > 0.9


def test_engine_runs_quantized_model():
    torch.manual_seed(0)
    model = LlamaForCausalLM(ModelConfig.tiny())
    quantize_model(model, QuantConfig("int8", min_features=1))
    eng = LLMEngine(model, ByteTokenizer(), EngineConfig(num_gpu_blocks=64, block_size=8, use_cuda_graphs=False,
                                                           device="cpu", max_model_len=128))
    outs = eng.generate(["hello", "quantized world"], SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
    assert len(outs) == 2 and all(len(o) > 0 for o in outs)


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
                    reason="needs Triton interpreter or a GPU")
def test_w8a16_gemm_kernel():
    from minivllm.quant.kernels import HAS_TRITON, w8a16_gemm

    if not HAS_TRITON:
        pytest.skip("triton not installed")
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16
    x = (torch.randn(5, 192, device=dev) * 0.5).to(dt)
    w = torch.randn(80, 192, device=dev) * 0.05
    wq, scale = quantize_weight(w, "int8")
    y = w8a16_gemm(x, wq.to(dev), scale.to(dev))
    ref = (x.float() @ dequantize_weight(wq.to(dev), scale.to(dev), torch.float32).T).to(dt)
    assert torch.allclose(y.float(), ref.float(), atol=2e-2, rtol=2e-2), (y.float() - ref.float()).abs().max()
