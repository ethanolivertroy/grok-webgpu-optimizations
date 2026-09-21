// Grok 4.7 Mamba-2 SSD decode (seq = 1).
//
// Same update as kernels/mamba2_decode.wgsl: dA = exp(softplus(dt) * A),
// h = dA * h + dt * B * x, y = dot(C, h) + D * x.
//
// 4.6 launched one 32-thread workgroup per (head, dim): 7680 workgroups per
// layer, each tree-reducing 128 state lanes. The state loop is 128 FMAs.
// That is less work than the launch and the barriers. 4.7 uses one thread
// per (head, dim) and walks the state as vec4s. 30 workgroups, no barrier.
// X, B, C, dt, and aux are read-only so the compiler can cache them.

struct Params {
  dt_min: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> C: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> dt: array<f32>;
@group(0) @binding(4) var<storage, read> aux: array<f32>;
@group(0) @binding(5) var<storage, read_write> H: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

const HEADS: u32 = 96u;
const HD: u32 = 80u;
const SSM: u32 = 128u;
const HPG: u32 = 12u;
const DT_BIAS: u32 = 0u;
const D_OFF: u32 = 7680u;
const A_OFF: u32 = 15360u;
const WG: u32 = 256u;

fn softplus(v: f32) -> f32 {
  if (v > 20.0) { return v; }
  return log(1.0 + exp(v));
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= HEADS * HD) { return; }
  let h = idx / HD;
  let d = idx % HD;
  let group = h / HPG;
  let xv = X[idx];
  let dt_s = max(softplus(dt[h] + aux[DT_BIAS + idx]), params.dt_min);
  let dA = exp(dt_s * aux[A_OFF + h]);
  let scale = dt_s * xv;
  let hBase = idx * (SSM / 4u);
  let bBase = group * (SSM / 4u);

  var y = 0.0;
  for (var i = 0u; i < (SSM / 4u); i++) {
    let hv = H[hBase + i] * dA + B[bBase + i] * scale;
    H[hBase + i] = hv;
    y += dot(C[bBase + i], hv);
  }
  Y[idx] = y + aux[D_OFF + idx] * xv;
}
