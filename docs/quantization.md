# Weight-only INT8 / FP8 quantization

Decode is bandwidth-bound: at batch size 1 every step streams the entire
weight set for one token. Storing weights in 8 bits halves the bytes moved,
which is where the ~2x decode speed-up at small batch comes from; activations
stay fp16/bf16 (W8A16), so no calibration set is needed.

## Format

Per-output-channel symmetric, one fp32 scale per row:

- **int8**: `w_q = round(w / s)`, `s = max|w| / 127` — any hardware
- **fp8 (E4M3)**: `w_q = (w / s).to(float8_e4m3fn)`, `s = max|w| / 448` —
  native on Ada / Hopper (sm_89+); same bytes as int8, better dynamic range
  for outlier channels

`quantize_model(model, QuantConfig("int8"))` swaps every projection for a
`QuantLinear` in place (`lm_head` is skipped by default — it is tied to the
embedding and sensitive to rounding). `QuantLinear` keeps the TP semantics of
the layer it replaced (all-reduce for row-parallel, all-gather for the vocab
projection).

## Kernel

`quant/kernels.py` is a Triton GEMM whose K loop loads an fp16 `x` tile and an
8-bit `w` tile, up-converts `w` in registers and feeds `tl.dot`; the per-row
scale is applied once to the fp32 accumulator after the loop. Global traffic
for the weight is therefore 1 byte/element. Tile shapes are picked by `M`:
decode (`M <= 16`) uses a 16x128 tile with a long K tile so each weight byte is
loaded once and amortised over the few rows; prefill uses 128x128.

On CPU / under `TRITON_INTERPRET=1` the layer falls back to
`F.linear(x, dequantize(w))`, which the kernel test compares against.

## Accuracy

`tests/test_quant.py` checks the round-trip error (< 1% relative for int8 on
Gaussian weights), that the quantized tiny model's logits stay within 5% of
the fp32 model's and that > 90% of argmax tokens agree, and that the engine
runs end-to-end on a quantized model.
