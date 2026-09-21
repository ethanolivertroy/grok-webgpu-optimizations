// CPU reference for the 4.7 q4 GEMV and Mamba SSD kernels.
// Nibble order matches kernels47/gemv_f32.wgsl: low nibble is the even weight.

export function loadZp(zpU32, col, nBlocks, blk) {
  const nib = col * nBlocks + blk;
  return (zpU32[nib >>> 3] >>> ((nib & 7) * 4)) & 15;
}

export function dequantDot(quant, scales, zpU32, col, nBlocks, a) {
  let acc = 0;
  for (let blk = 0; blk < nBlocks; blk++) {
    const scale = scales[col * nBlocks + blk];
    const zero = loadZp(zpU32, col, nBlocks, blk);
    const base = (col * nBlocks + blk) * 16;
    for (let i = 0; i < 32; i++) {
      const byte = quant[base + (i >> 1)];
      const nib = (i & 1) ? (byte >>> 4) & 15 : byte & 15;
      acc += (nib - zero) * scale * a[blk * 32 + i];
    }
  }
  return acc;
}

export function gemvRef(quant, scales, zpU32, a, N, nBlocks, epilogue = 0) {
  const y = new Float32Array(N);
  for (let col = 0; col < N; col++) {
    let v = dequantDot(quant, scales, zpU32, col, nBlocks, a);
    if (epilogue === 1) {
      const r = Math.max(v, 0);
      v = r * r;
    }
    y[col] = v;
  }
  return y;
}

function softplus(v) {
  if (v > 20) return v;
  return Math.log(1 + Math.exp(v));
}

// In-place on H. Shapes match Nemotron Nano Mamba-2 decode.
export function ssdRef({ x, B, C, dt, aux, H, dtMin }) {
  const HD = 80;
  const HEADS = 96;
  const N = 128;
  const HPG = 12;
  const DT_BIAS = 0;
  const D_OFF = 7680;
  const A_OFF = 15360;
  const Y = new Float32Array(HEADS * HD);
  for (let h = 0; h < HEADS; h++) {
    const Av = aux[A_OFF + h];
    const group = Math.floor(h / HPG);
    for (let d = 0; d < HD; d++) {
      const idx = h * HD + d;
      const xv = x[idx];
      const dtS = Math.max(softplus(dt[h] + aux[DT_BIAS + idx]), dtMin);
      const dA = Math.exp(dtS * Av);
      const hOff = idx * N;
      const bOff = group * N;
      let y = 0;
      for (let n = 0; n < N; n++) {
        const hv = H[hOff + n] * dA + dtS * B[bOff + n] * xv;
        H[hOff + n] = hv;
        y += C[bOff + n] * hv;
      }
      Y[idx] = y + aux[D_OFF + idx] * xv;
    }
  }
  return Y;
}
