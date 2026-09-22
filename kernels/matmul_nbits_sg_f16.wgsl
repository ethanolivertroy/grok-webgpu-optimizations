// sg4 GEMV with f16 inner products. Same q4 packing and zp as matmul_nbits_sg.wgsl.
// A is converted to f16 once per K block. All four column vec4 loads run before unpack.

enable subgroups;
enable f16;

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

const WG: u32 = 256u;
const N_COLS: u32 = 4u;

var<workgroup> sg_partials: array<vec4<f32>, 8>;

fn load_zp(col: u32, blk: u32) -> f32 {
  let nib = col * params.n_blocks + blk;
  return f32((zp[nib / 8u] >> ((nib % 8u) * 4u)) & 15u);
}

fn dequant_word(packed: u32, s: vec4<f16>, z: vec4<f16>) -> array<vec4<f16>, 2> {
  let lo = unpack4xU8(packed & 0x0F0F0F0Fu);
  let hi = unpack4xU8((packed >> 4u) & 0x0F0F0F0Fu);
  let d0 = (vec4<f16>(f16(lo.x), f16(hi.x), f16(lo.y), f16(hi.y)) - z) * s;
  let d1 = (vec4<f16>(f16(lo.z), f16(hi.z), f16(lo.w), f16(hi.w)) - z) * s;
  return array<vec4<f16>, 2>(d0, d1);
}

fn as_f16(v: vec4<f32>) -> vec4<f16> {
  return vec4<f16>(f16(v.x), f16(v.y), f16(v.z), f16(v.w));
}

fn relu2(v: f32) -> f32 {
  let r = max(v, 0.0);
  return r * r;
}

fn dequant_dot(packed: vec4<u32>, scale: f32, zero: f32, a0: vec4<f16>, a1: vec4<f16>, a2: vec4<f16>, a3: vec4<f16>, a4: vec4<f16>, a5: vec4<f16>, a6: vec4<f16>, a7: vec4<f16>) -> f32 {
  let s = vec4<f16>(f16(scale));
  let z = vec4<f16>(f16(zero));
  var acc: f16 = 0.0h;
  {
    let w = dequant_word(packed.x, s, z);
    acc += dot(a0, w[0]) + dot(a1, w[1]);
  }
  {
    let w = dequant_word(packed.y, s, z);
    acc += dot(a2, w[0]) + dot(a3, w[1]);
  }
  {
    let w = dequant_word(packed.z, s, z);
    acc += dot(a4, w[0]) + dot(a5, w[1]);
  }
  {
    let w = dequant_word(packed.w, s, z);
    acc += dot(a6, w[0]) + dot(a7, w[1]);
  }
  return f32(acc);
}

fn sg_sum_v4(value: vec4<f32>) -> vec4<f32> {
  var x = value;
  x = x + subgroupShuffleXor(x, 1u);
  x = x + subgroupShuffleXor(x, 2u);
  x = x + subgroupShuffleXor(x, 4u);
  x = x + subgroupShuffleXor(x, 8u);
  x = x + subgroupShuffleXor(x, 16u);
  return x;
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
  let live0 = c0 < n;
  let live1 = c1 < n;
  let live2 = c2 < n;
  let live3 = c3 < n;
  var acc = vec4<f32>(0.0);

  for (var blk = tid; blk < n_blocks; blk += WG) {
    let a_base = blk * 8u;
    let a0 = as_f16(A[a_base + 0u]);
    let a1 = as_f16(A[a_base + 1u]);
    let a2 = as_f16(A[a_base + 2u]);
    let a3 = as_f16(A[a_base + 3u]);
    let a4 = as_f16(A[a_base + 4u]);
    let a5 = as_f16(A[a_base + 5u]);
    let a6 = as_f16(A[a_base + 6u]);
    let a7 = as_f16(A[a_base + 7u]);

    var p0 = vec4<u32>(0u);
    var p1 = vec4<u32>(0u);
    var p2 = vec4<u32>(0u);
    var p3 = vec4<u32>(0u);
    if (live0) { p0 = B[c0 * n_blocks + blk]; }
    if (live1) { p1 = B[c1 * n_blocks + blk]; }
    if (live2) { p2 = B[c2 * n_blocks + blk]; }
    if (live3) { p3 = B[c3 * n_blocks + blk]; }

    if (live0) {
      acc.x += dequant_dot(p0, scales[c0 * n_blocks + blk], load_zp(c0, blk), a0, a1, a2, a3, a4, a5, a6, a7);
    }
    if (live1) {
      acc.y += dequant_dot(p1, scales[c1 * n_blocks + blk], load_zp(c1, blk), a0, a1, a2, a3, a4, a5, a6, a7);
    }
    if (live2) {
      acc.z += dequant_dot(p2, scales[c2 * n_blocks + blk], load_zp(c2, blk), a0, a1, a2, a3, a4, a5, a6, a7);
    }
    if (live3) {
      acc.w += dequant_dot(p3, scales[c3 * n_blocks + blk], load_zp(c3, blk), a0, a1, a2, a3, a4, a5, a6, a7);
    }
  }

  let red = sg_sum_v4(acc);
  if ((tid & 31u) == 0u) {
    sg_partials[tid >> 5u] = red;
  }
  workgroupBarrier();
  if (tid == 0u) {
    var tot = vec4<f32>(0.0);
    for (var i = 0u; i < 8u; i++) {
      tot += sg_partials[i];
    }
    if (params.epilogue == 1u) {
      tot = vec4<f32>(relu2(tot.x), relu2(tot.y), relu2(tot.z), relu2(tot.w));
    }
    if (live0) { Y[c0] = tot.x; }
    if (live1) { Y[c1] = tot.y; }
    if (live2) { Y[c2] = tot.z; }
    if (live3) { Y[c3] = tot.w; }
  }
}
