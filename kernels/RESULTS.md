# Measured results (M5 Max)

MacBook Pro M5 Max, 64 GB, Chrome 151, WebGPU to Metal
(`vendor: apple`, `architecture: metal-3`, `shaderF16: true`).
Gate prompt uses the instruct chat template. 128 new tokens, greedy.
Do not mix these with software-adapter numbers.

## Full-model decode

| Mode | tok/s | ms | notes |
|------|-------|-----|-------|
| stock ORT | **69.76** | 1834.9 | transformers.js + ORT WebGPU |
| ORT shader-body swap | 71.30 | 1795.3 | +2.2%, ORT still owns dispatch |
| fused engine, first loop | 100.47 | 1274.1 | owned GEMV + fused Mamba/attn |
| one-pass + RW activations | 95.94 | 1334.2 | regression: GEMV A as read_write |
| + parallel Mamba SSD, sg32 | 116.63 | 1097.5 | GPU 141 tok/s |
| + GPU-resident next token | 125.45 | 1020.4 | GPU 140 tok/s |
| + f16 GEMV mlp_up / mlp_down | 129.11 | 991.4 | raw 8-token encode, not chat template |
| + chat template (24 prompt ids) | **115.04** | 1112.7 | GPU 142 tok/s, first token matches ORT |

Chat streaming with overlapped readback measures about 112 tok/s wall on the
same machine. The 129 figure is the earlier raw-encode iteration. Do not treat
it as the chat-template number.

Raw: `harness/metal/m5max-decode-chat-template.json`.

## GEMV microbench (lm_head 3136 x 131072 q4)

| Kernel | ms | vs ORT tile |
|--------|-----|-------------|
| ORT tile (WG=128, 8 cols) | 0.634 | 1.00x |
| sg4 (WG=256, 4 cols) | 0.488 | 1.30x |
| f16 inner product | 0.488 | 1.30x |
| sg32 (WG=32, 8 cols) | 0.500 | 1.27x |
| subgroup-matrix 8x8 | 2.642 | 0.24x |

f16 wins mlp_up and mlp_down. Those two shapes use the f16 kernel. lm_head stays sg4.

Raw: `harness/metal/m5max-engine-gemv-f16.json`.

Written by Grok 4.6.

## Grok 4.7 follow-up (2026-09-27)

Same compare page, chat template, 128 greedy tokens, Chrome to Metal on the
M5 Max (128 GB). `token_agree` is how many new token ids match 4.6.

### What already won

Pull request #3 kept the 4.6 GEMV grids and the faster Mamba SSD launch.
On that compare, 4.7 was **113.36** wall tok/s and **140.04** GPU tok/s.
Same-run 4.6 was 108.92 wall and 133.8 GPU. Token ids matched.

A blanket 64-column GEMV was slower: 4.7 at 90.77 wall vs 4.6 at 105.71,
tokens matched. MLP lost. Mamba in_proj was flat. Leave that grid out.

### What did not win

Three commits on top of #3:

- #4 Hoist. The f16 MLP converts A once per K block and issues all four
  column `vec4` weight loads before unpack. sg32 loads A once and issues
  all eight column loads before unpack.
- #5 Unroll. MLP and in_proj K loops take two blocks: load both, add the
  first, then add the second. A leftover block uses the one-block body.
- #6 Type fix. Merging #5 and then #4 left the MLP hot loop passing
  `vec4<f32>` into `dequant_dot`, which takes `vec4<f16>`. Both blocks
  now convert with `as_f16` once, before the dots.

Workgroups stayed 256x4 (`kernels/matmul_nbits_sg_f16.wgsl`) and 32x8
(`kernels/matmul_nbits_sg32.wgsl`). The interleaved q4 dot stayed: low
nibble first, `d0 = (lo.x, hi.x, lo.y, hi.y)`, `d1 = (lo.z, hi.z, lo.w, hi.w)`,
`dot(a0, d0) + dot(a1, d1)`. The f16 kernel still accumulates a block in
f16 and converts to f32 once per column per block.

Keep that unpack. An even/odd lane dot holds the same nibble values and
changes f16 add order, so token agreement is unproven.

Measured on main `266151b2ffb4277fb2cfd378470e798df3796bd2`, which includes
#4, #5, #6, and the Dependabot npm bump. Google Chrome,
`--use-angle=metal`. Two runs, back to back.

| Run | 4.7 wall | 4.7 GPU | 4.6 wall | 4.6 GPU | token_agree |
|-----|----------|---------|----------|---------|-------------|
| 1 | 88.93 | 109.04 | 91.29 | 112.21 | 128 |
| 2 | 86.89 | 106.27 | 86.84 | 106.13 | 128 |

Mean 4.7: **87.91** wall, **107.66** GPU. Run 1 is behind 4.6 on both
clocks. Run 2 is a tie (0.05 wall, 0.14 GPU).

The page still publishes the 4.6 chat-template mark, 115.04 wall and
142.48 GPU. Tonight's live 4.6 sat in the same lower band as 4.7, so the
session was slower than that published mark for both engines. Judge a
kernel by the 4.6 number in the same run.

Raw: `harness/metal/m5max-47-hoist-unroll-compare.json`.

### For the next attempt

Decode GEMV is limited by weight loads. Hoist and unroll still read every
packed byte. They only change when the activation convert and the column
loads happen. Tok/s stays flat.

Leave these in place:

- 256x4 f16 MLP, 32x8 sg32 in_proj, sg4 for the other projections
- interleaved nibble order, and f16 accumulation with one f32 convert per column per block
- the one-thread-per-element Mamba SSD launch from #3

A faster pass has to move less weight, or fuse launches that reread the
same activations. Another reshape of this dot will not clear 113.36 wall
and 140.04 GPU.

On a Mac the bench script's default Chrome directory is
`/root/.cache/ms-playwright`. Set `CHROME_PATH`.

```bash
CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
node scripts/headless-bench.mjs --page compare --tokens 128 --angle metal
```

Pass when `token_agree` is 128, 4.7 beats that run's 4.6 on wall and GPU,
and the mean is above 113.36 wall and 140.04 GPU.

Written by Grok 4.7, for the next pass.
