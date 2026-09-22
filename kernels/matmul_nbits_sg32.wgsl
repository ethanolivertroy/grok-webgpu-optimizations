// Apple-tuned decode GEMV: one 32-wide subgroup per 8 N-columns.
// q4 block=32. Zero-points packed 8 per u32. Dispatch ceil(N/8).
// epilogue=1 applies relu2 on the written columns.
// A is loaded once per K block. All eight column vec4 loads run before unpack.

enable subgroups;

struct Params {
  K: u32,
  N: u32,
  n_blocks: u32,
  epilogue: u32,
};

@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> zp: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const WG: u32 = 32u;
const N_COLS: u32 = 8u;

fn load_zp(col: u32, blk: u32) -> f32 {
  let nib = col * params.n_blocks + blk;
  return f32((zp[nib / 8u] >> ((nib % 8u) * 4u)) & 15u);
}

fn dequant_dot(packed: vec4<u32>, scale: f32, zero: f32, a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>, a4: vec4<f32>, a5: vec4<f32>, a6: vec4<f32>, a7: vec4<f32>) -> f32 {
  let s = vec4<f32>(scale);
  let z = vec4<f32>(zero);
  var acc: f32 = 0.0;
  {
    let lo = unpack4xU8(packed.x & 0x0F0F0F0Fu);
    let hi = unpack4xU8((packed.x >> 4u) & 0x0F0F0F0Fu);
    let d0 = (vec4<f32>(f32(lo.x), f32(hi.x), f32(lo.y), f32(hi.y)) - z) * s;
    let d1 = (vec4<f32>(f32(lo.z), f32(hi.z), f32(lo.w), f32(hi.w)) - z) * s;
    acc += dot(a0, d0) + dot(a1, d1);
  }
  {
    let lo = unpack4xU8(packed.y & 0x0F0F0F0Fu);
    let hi = unpack4xU8((packed.y >> 4u) & 0x0F0F0F0Fu);
    let d0 = (vec4<f32>(f32(lo.x), f32(hi.x), f32(lo.y), f32(hi.y)) - z) * s;
    let d1 = (vec4<f32>(f32(lo.z), f32(hi.z), f32(lo.w), f32(hi.w)) - z) * s;
    acc += dot(a2, d0) + dot(a3, d1);
  }
  {
    let lo = unpack4xU8(packed.z & 0x0F0F0F0Fu);
    let hi = unpack4xU8((packed.z >> 4u) & 0x0F0F0F0Fu);
    let d0 = (vec4<f32>(f32(lo.x), f32(hi.x), f32(lo.y), f32(hi.y)) - z) * s;
    let d1 = (vec4<f32>(f32(lo.z), f32(hi.z), f32(lo.w), f32(hi.w)) - z) * s;
    acc += dot(a4, d0) + dot(a5, d1);
  }
  {
    let lo = unpack4xU8(packed.w & 0x0F0F0F0Fu);
    let hi = unpack4xU8((packed.w >> 4u) & 0x0F0F0F0Fu);
    let d0 = (vec4<f32>(f32(lo.x), f32(hi.x), f32(lo.y), f32(hi.y)) - z) * s;
    let d1 = (vec4<f32>(f32(lo.z), f32(hi.z), f32(lo.w), f32(hi.w)) - z) * s;
    acc += dot(a6, d0) + dot(a7, d1);
  }
  return acc;
}

fn sg_sum(value: f32) -> f32 {
  var x = value;
  x = x + subgroupShuffleXor(x, 1u);
  x = x + subgroupShuffleXor(x, 2u);
  x = x + subgroupShuffleXor(x, 4u);
  x = x + subgroupShuffleXor(x, 8u);
  x = x + subgroupShuffleXor(x, 16u);
  return x;
}

fn relu2(v: f32) -> f32 {
  let r = max(v, 0.0);
  return r * r;
}

@compute @workgroup_size(WG, 1, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  let col0 = wg.x * N_COLS;
  let n_blocks = params.n_blocks;
  let n = params.N;
  let c0 = col0 + 0u;
  let c1 = col0 + 1u;
  let c2 = col0 + 2u;
  let c3 = col0 + 3u;
  let c4 = col0 + 4u;
  let c5 = col0 + 5u;
  let c6 = col0 + 6u;
  let c7 = col0 + 7u;
  let live0 = c0 < n;
  let live1 = c1 < n;
  let live2 = c2 < n;
  let live3 = c3 < n;
  let live4 = c4 < n;
  let live5 = c5 < n;
  let live6 = c6 < n;
  let live7 = c7 < n;
  var acc: array<f32, 8>;
  for (var r = 0u; r < N_COLS; r++) {
    acc[r] = 0.0;
  }

  for (var blk = tid; blk < n_blocks; blk += WG) {
    let a_base = blk * 8u;
    let a0 = A[a_base + 0u];
    let a1 = A[a_base + 1u];
    let a2 = A[a_base + 2u];
    let a3 = A[a_base + 3u];
    let a4 = A[a_base + 4u];
    let a5 = A[a_base + 5u];
    let a6 = A[a_base + 6u];
    let a7 = A[a_base + 7u];

    var p0 = vec4<u32>(0u);
    var p1 = vec4<u32>(0u);
    var p2 = vec4<u32>(0u);
    var p3 = vec4<u32>(0u);
    var p4 = vec4<u32>(0u);
    var p5 = vec4<u32>(0u);
    var p6 = vec4<u32>(0u);
    var p7 = vec4<u32>(0u);
    if (live0) { p0 = B[c0 * n_blocks + blk]; }
    if (live1) { p1 = B[c1 * n_blocks + blk]; }
    if (live2) { p2 = B[c2 * n_blocks + blk]; }
    if (live3) { p3 = B[c3 * n_blocks + blk]; }
    if (live4) { p4 = B[c4 * n_blocks + blk]; }
    if (live5) { p5 = B[c5 * n_blocks + blk]; }
    if (live6) { p6 = B[c6 * n_blocks + blk]; }
    if (live7) { p7 = B[c7 * n_blocks + blk]; }

    if (live0) { acc[0] += dequant_dot(p0, scales[c0 * n_blocks + blk], load_zp(c0, blk), a0, a1, a2, a3, a4, a5, a6, a7); }
    if (live1) { acc[1] += dequant_dot(p1, scales[c1 * n_blocks + blk], load_zp(c1, blk), a0, a1, a2, a3, a4, a5, a6, a7); }
    if (live2) { acc[2] += dequant_dot(p2, scales[c2 * n_blocks + blk], load_zp(c2, blk), a0, a1, a2, a3, a4, a5, a6, a7); }
    if (live3) { acc[3] += dequant_dot(p3, scales[c3 * n_blocks + blk], load_zp(c3, blk), a0, a1, a2, a3, a4, a5, a6, a7); }
    if (live4) { acc[4] += dequant_dot(p4, scales[c4 * n_blocks + blk], load_zp(c4, blk), a0, a1, a2, a3, a4, a5, a6, a7); }
    if (live5) { acc[5] += dequant_dot(p5, scales[c5 * n_blocks + blk], load_zp(c5, blk), a0, a1, a2, a3, a4, a5, a6, a7); }
    if (live6) { acc[6] += dequant_dot(p6, scales[c6 * n_blocks + blk], load_zp(c6, blk), a0, a1, a2, a3, a4, a5, a6, a7); }
    if (live7) { acc[7] += dequant_dot(p7, scales[c7 * n_blocks + blk], load_zp(c7, blk), a0, a1, a2, a3, a4, a5, a6, a7); }
  }

  for (var r = 0u; r < N_COLS; r++) {
    let tot0 = sg_sum(acc[r]);
    let col = col0 + r;
    if (tid == 0u && col < n) {
      var tot = tot0;
      if (params.epilogue == 1u) {
        tot = relu2(tot);
      }
      Y[col] = tot;
    }
  }
}
