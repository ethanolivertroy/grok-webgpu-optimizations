// Apple-tuned decode GEMV: one 32-wide subgroup per 8 N-columns.
// q4 block=32. Zero-points packed 8 per u32. Dispatch ceil(N/8).
// epilogue=1 applies relu2 on the written columns.

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
  var acc: array<f32, 8>;
  for (var r = 0u; r < N_COLS; r++) {
    acc[r] = 0.0;
  }

  var blk = tid;
  let step = WG;
  while (blk + step < n_blocks) {
    let blk1 = blk + step;
    let a_base = blk * 8u;
    let b_base = blk1 * 8u;
    let a0 = A[a_base + 0u];
    let a1 = A[a_base + 1u];
    let a2 = A[a_base + 2u];
    let a3 = A[a_base + 3u];
    let a4 = A[a_base + 4u];
    let a5 = A[a_base + 5u];
    let a6 = A[a_base + 6u];
    let a7 = A[a_base + 7u];
    let b0 = A[b_base + 0u];
    let b1 = A[b_base + 1u];
    let b2 = A[b_base + 2u];
    let b3 = A[b_base + 3u];
    let b4 = A[b_base + 4u];
    let b5 = A[b_base + 5u];
    let b6 = A[b_base + 6u];
    let b7 = A[b_base + 7u];
    var p0: array<vec4<u32>, 8>;
    var s0: array<f32, 8>;
    var z0: array<f32, 8>;
    var p1: array<vec4<u32>, 8>;
    var s1: array<f32, 8>;
    var z1: array<f32, 8>;
    for (var r = 0u; r < N_COLS; r++) {
      let col = col0 + r;
      if (col < n) {
        p0[r] = B[col * n_blocks + blk];
        s0[r] = scales[col * n_blocks + blk];
        z0[r] = load_zp(col, blk);
        p1[r] = B[col * n_blocks + blk1];
        s1[r] = scales[col * n_blocks + blk1];
        z1[r] = load_zp(col, blk1);
      }
    }
    for (var r = 0u; r < N_COLS; r++) {
      let col = col0 + r;
      if (col < n) {
        acc[r] += dequant_dot(p0[r], s0[r], z0[r], a0, a1, a2, a3, a4, a5, a6, a7);
      }
    }
    for (var r = 0u; r < N_COLS; r++) {
      let col = col0 + r;
      if (col < n) {
        acc[r] += dequant_dot(p1[r], s1[r], z1[r], b0, b1, b2, b3, b4, b5, b6, b7);
      }
    }
    blk += step * 2u;
  }
  if (blk < n_blocks) {
    let a_base = blk * 8u;
    let a0 = A[a_base + 0u];
    let a1 = A[a_base + 1u];
    let a2 = A[a_base + 2u];
    let a3 = A[a_base + 3u];
    let a4 = A[a_base + 4u];
    let a5 = A[a_base + 5u];
    let a6 = A[a_base + 6u];
    let a7 = A[a_base + 7u];
    for (var r = 0u; r < N_COLS; r++) {
      let col = col0 + r;
      if (col < n) {
        let packed = B[col * n_blocks + blk];
        let scale = scales[col * n_blocks + blk];
        acc[r] += dequant_dot(packed, scale, load_zp(col, blk), a0, a1, a2, a3, a4, a5, a6, a7);
      }
    }
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
