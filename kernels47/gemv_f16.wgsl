// Grok 4.7 f16 decode GEMV. Same schedule as gemv_f32.wgsl.
// Per-block dot is f16. The sum across blocks stays f32, matching the 4.6
// MLP kernel. relu2 (epilogue 1) runs on that f32 sum. MLP up and down only.

enable subgroups;
enable f16;

struct Params {
  N: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

override N_BLOCKS: u32 = 98u;
override EPILOGUE: u32 = 0u;

@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> zp: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const WG: u32 = 256u;
const COLS: u32 = 8u;

fn load_zp(col: u32, blk: u32) -> f32 {
  let nib = col * N_BLOCKS + blk;
  return f32((zp[nib / 8u] >> ((nib % 8u) * 4u)) & 15u);
}

fn as_f16(v: vec4<f32>) -> vec4<f16> {
  return vec4<f16>(f16(v.x), f16(v.y), f16(v.z), f16(v.w));
}

fn dequant_word(packed: u32, s: vec4<f16>, z: vec4<f16>) -> array<vec4<f16>, 2> {
  let lo = unpack4xU8(packed & 0x0F0F0F0Fu);
  let hi = unpack4xU8((packed >> 4u) & 0x0F0F0F0Fu);
  let d0 = (vec4<f16>(f16(lo.x), f16(hi.x), f16(lo.y), f16(hi.y)) - z) * s;
  let d1 = (vec4<f16>(f16(lo.z), f16(hi.z), f16(lo.w), f16(hi.w)) - z) * s;
  return array<vec4<f16>, 2>(d0, d1);
}

fn dequant_dot(packed: vec4<u32>, scale: f32, zero: f32, a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>, a4: vec4<f32>, a5: vec4<f32>, a6: vec4<f32>, a7: vec4<f32>) -> f32 {
  let s = vec4<f16>(f16(scale));
  let z = vec4<f16>(f16(zero));
  var acc: f16 = 0.0h;
  {
    let w = dequant_word(packed.x, s, z);
    acc += dot(as_f16(a0), w[0]) + dot(as_f16(a1), w[1]);
  }
  {
    let w = dequant_word(packed.y, s, z);
    acc += dot(as_f16(a2), w[0]) + dot(as_f16(a3), w[1]);
  }
  {
    let w = dequant_word(packed.z, s, z);
    acc += dot(as_f16(a4), w[0]) + dot(as_f16(a5), w[1]);
  }
  {
    let w = dequant_word(packed.w, s, z);
    acc += dot(as_f16(a6), w[0]) + dot(as_f16(a7), w[1]);
  }
  return f32(acc);
}

fn relu2(v: f32) -> f32 {
  let r = max(v, 0.0);
  return r * r;
}

fn col_dot(col: u32, blk: u32, a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>, a4: vec4<f32>, a5: vec4<f32>, a6: vec4<f32>, a7: vec4<f32>) -> f32 {
  if (col >= params.N) {
    return 0.0;
  }
  let packed = B[col * N_BLOCKS + blk];
  let scale = scales[col * N_BLOCKS + blk];
  return dequant_dot(packed, scale, load_zp(col, blk), a0, a1, a2, a3, a4, a5, a6, a7);
}

@compute @workgroup_size(WG, 1, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(subgroup_id) sg: u32,
  @builtin(subgroup_invocation_id) lane: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let nsg = WG / sg_size;
  let col0 = (wg.x * nsg + sg) * COLS;
  var acc0 = 0.0;
  var acc1 = 0.0;
  var acc2 = 0.0;
  var acc3 = 0.0;
  var acc4 = 0.0;
  var acc5 = 0.0;
  var acc6 = 0.0;
  var acc7 = 0.0;

  let n_iter = (N_BLOCKS + sg_size - 1u) / sg_size;
  for (var i = 0u; i < n_iter; i++) {
    let blk_i = lane + i * sg_size;
    let live = blk_i < N_BLOCKS;
    let blk = select(0u, blk_i, live);
    let base = blk * 8u;
    let a0 = A[base + 0u];
    let a1 = A[base + 1u];
    let a2 = A[base + 2u];
    let a3 = A[base + 3u];
    let a4 = A[base + 4u];
    let a5 = A[base + 5u];
    let a6 = A[base + 6u];
    let a7 = A[base + 7u];
    let d0 = col_dot(col0 + 0u, blk, a0, a1, a2, a3, a4, a5, a6, a7);
    let d1 = col_dot(col0 + 1u, blk, a0, a1, a2, a3, a4, a5, a6, a7);
    let d2 = col_dot(col0 + 2u, blk, a0, a1, a2, a3, a4, a5, a6, a7);
    let d3 = col_dot(col0 + 3u, blk, a0, a1, a2, a3, a4, a5, a6, a7);
    let d4 = col_dot(col0 + 4u, blk, a0, a1, a2, a3, a4, a5, a6, a7);
    let d5 = col_dot(col0 + 5u, blk, a0, a1, a2, a3, a4, a5, a6, a7);
    let d6 = col_dot(col0 + 6u, blk, a0, a1, a2, a3, a4, a5, a6, a7);
    let d7 = col_dot(col0 + 7u, blk, a0, a1, a2, a3, a4, a5, a6, a7);
    acc0 += select(0.0, d0, live);
    acc1 += select(0.0, d1, live);
    acc2 += select(0.0, d2, live);
    acc3 += select(0.0, d3, live);
    acc4 += select(0.0, d4, live);
    acc5 += select(0.0, d5, live);
    acc6 += select(0.0, d6, live);
    acc7 += select(0.0, d7, live);
  }

  let red0 = subgroupAdd(vec4<f32>(acc0, acc1, acc2, acc3));
  let red1 = subgroupAdd(vec4<f32>(acc4, acc5, acc6, acc7));
  if (lane == 0u) {
    var s0 = red0.x;
    var s1 = red0.y;
    var s2 = red0.z;
    var s3 = red0.w;
    var s4 = red1.x;
    var s5 = red1.y;
    var s6 = red1.z;
    var s7 = red1.w;
    if (EPILOGUE == 1u) {
      s0 = relu2(s0);
      s1 = relu2(s1);
      s2 = relu2(s2);
      s3 = relu2(s3);
      s4 = relu2(s4);
      s5 = relu2(s5);
      s6 = relu2(s6);
      s7 = relu2(s7);
    }
    let n = params.N;
    if (col0 + 0u < n) { Y[col0 + 0u] = s0; }
    if (col0 + 1u < n) { Y[col0 + 1u] = s1; }
    if (col0 + 2u < n) { Y[col0 + 2u] = s2; }
    if (col0 + 3u < n) { Y[col0 + 3u] = s3; }
    if (col0 + 4u < n) { Y[col0 + 4u] = s4; }
    if (col0 + 5u < n) { Y[col0 + 5u] = s5; }
    if (col0 + 6u < n) { Y[col0 + 6u] = s6; }
    if (col0 + 7u < n) { Y[col0 + 7u] = s7; }
  }
}
