import {
  compileCompute,
  createEmptyStorage,
  createUniform,
} from '../kernels/gpu.js';
import { NANO4B, LAYER_TYPES } from '../engine/config.js';
import { q4Meta } from '../engine/weights.js';
import gemvWgsl from '../kernels/matmul_nbits_sg.wgsl?raw';
import gemv32Wgsl from '../kernels/matmul_nbits_sg32.wgsl?raw';
import gemvF16Wgsl from '../kernels/matmul_nbits_sg_f16.wgsl?raw';
import ssdWgsl from '../kernels47/mamba_ssd.wgsl?raw';
import rmsWgsl from '../kernels/rms.wgsl?raw';
import addWgsl from '../kernels/add.wgsl?raw';
import convWgsl from '../kernels/conv1d_update.wgsl?raw';
import gatedWgsl from '../kernels/gated_group_rms.wgsl?raw';
import kvWgsl from '../kernels/kv_append.wgsl?raw';
import attnWgsl from '../kernels/sparse_attn.wgsl?raw';
import embedWgsl from '../kernels/embed_q4.wgsl?raw';
import argmaxPartialWgsl from '../kernels/argmax_partial.wgsl?raw';
import argmaxMergeWgsl from '../kernels/argmax_merge.wgsl?raw';

const C = NANO4B;
const INNER = C.mambaHeads * C.mambaHeadDim;

function uniformU32(device, vals) {
  return createUniform(device, Uint32Array.from(vals));
}

function uniformMixed(device, pairs) {
  const buf = new ArrayBuffer(Math.max(16, pairs.length * 4));
  const dv = new DataView(buf);
  pairs.forEach(([kind, v], i) => {
    if (kind === 'f32') dv.setFloat32(i * 4, v, true);
    else dv.setUint32(i * 4, v, true);
  });
  return createUniform(device, new Uint8Array(buf));
}

function bg(device, pipeline, resources, label) {
  return device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(0),
    entries: resources.map((r, i) => {
      const resource = r && r.buffer ? r : { buffer: r };
      if (!resource.buffer) {
        throw new Error(`bind ${label}[${i}]: missing buffer`);
      }
      return { binding: i, resource };
    }),
  });
}

function slice(buffer, offsetBytes, sizeBytes) {
  return { buffer, offset: offsetBytes, size: sizeBytes };
}

function gemvName(layer, kind) {
  return `model_layers_${layer}_${kind}_MatMul_weight`;
}

async function shaderModule(device, code, label) {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    const text = errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n');
    throw new Error(`WGSL compile failed (${label}):\n${text}`);
  }
  return module;
}

export class DecodeEngine47 {
  constructor(gpu, weights) {
    this.gpu = gpu;
    this.device = gpu.device;
    this.W = weights.buffers;
    this.meta = weights.meta;
    this.seq = 0;
    this.label = 'Grok 4.7';
  }

  static async create(gpu, weights) {
    const eng = new DecodeEngine47(gpu, weights);
    await eng._compile();
    eng._alloc();
    eng._bind();
    return eng;
  }

  q4(name) {
    const quant = this.W.get(`${name}_quant`);
    const scales = this.W.get(`${name}_scales`);
    const zp = this.W.get(`${name}_zp`);
    if (!quant || !scales) throw new Error(`q4 missing ${name}`);
    if (!zp) throw new Error(`q4 zp missing ${name}`);
    return { quant, scales, zp, ...q4Meta(this.meta, `${name}_quant`) };
  }

  async _compile() {
    const d = this.device;
    const f16 = d.features.has('shader-f16');
    const compileOne = async (code, label) => {
      const module = await shaderModule(d, code, label);
      return d.createComputePipeline({ label, layout: 'auto', compute: { module, entryPoint: 'main' } });
    };
    // GEMV grids are the 4.6 per-shape picks. The 64-column schedule lost on
    // MLP and on the sg4 projections, and did not beat sg32 on mamba in_proj.
    this.p = {
      f16,
      gemv: (await compileCompute(d, gemvWgsl, 'gemv-sg4')).pipeline,
      gemv32: (await compileCompute(d, gemv32Wgsl, 'gemv-sg32')).pipeline,
      gemvF16: f16 ? (await compileCompute(d, gemvF16Wgsl, 'gemv-sg4-f16')).pipeline : null,
      rms: await compileOne(rmsWgsl, 'rms'),
      add: await compileOne(addWgsl, 'add'),
      conv: await compileOne(convWgsl, 'conv1d'),
      mamba: await compileOne(ssdWgsl, 'mamba2-decode-47'),
      gated: await compileOne(gatedWgsl, 'gated-rms'),
      kv: await compileOne(kvWgsl, 'kv-append'),
      attn: await compileOne(attnWgsl, 'gqa'),
      embed: await compileOne(embedWgsl, 'embed-q4'),
      argmax0: await compileOne(argmaxPartialWgsl, 'argmax-partial'),
      argmax1: await compileOne(argmaxMergeWgsl, 'argmax-merge'),
    };
    this.gemvSchedule = 'f16-sg4 mlp, sg32 in_proj, sg4 else';
  }

  _alloc() {
    const d = this.device;
    const f = (n) => createEmptyStorage(d, n * 4);
    this.h = f(C.hidden);
    this.hNorm = f(C.hidden);
    this.mixer = f(C.hidden);
    this.mambaIn = f(C.mambaIn);
    this.convOut = f(C.convDim);
    this.ssdY = f(INNER);
    this.gated = f(INNER);
    this.mlp = f(C.intermediate);
    this.q = f(C.heads * C.headDim);
    this.k = f(C.kvHeads * C.headDim);
    this.v = f(C.kvHeads * C.headDim);
    this.attnO = f(C.heads * C.headDim);
    this.logits = f(C.vocab);
    this.tokenOut = d.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.tokenLog = d.createBuffer({
      size: C.maxSeq * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.tokenStaging = d.createBuffer({
      size: C.maxSeq * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.tsLog = d.createBuffer({
      size: C.maxSeq * 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.tsStaging = d.createBuffer({
      size: C.maxSeq * 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.argmaxPartial = createEmptyStorage(d, 256 * 2 * 4);
    this.streamStaging = [0, 1].map(() =>
      d.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
    );
    this.hasTs = d.features.has('timestamp-query');
    this.tsSet = this.hasTs ? d.createQuerySet({ type: 'timestamp', count: 2 }) : null;
    this.tsResolve = this.hasTs
      ? d.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
      : null;
    this.gpuNs = 0;
    this.gpuSamples = 0;

    this.convState = [];
    this.ssmState = [];
    this.kCache = [];
    this.vCache = [];
    for (let i = 0; i < C.layers; i++) {
      const t = LAYER_TYPES[i];
      if (t === 'mamba') {
        this.convState[i] = f(C.convDim * C.convKernel);
        this.ssmState[i] = f(C.mambaHeads * C.mambaHeadDim * C.ssmState);
      }
      if (t === 'attention') {
        this.kCache[i] = f(C.maxSeq * C.kvHeads * C.headDim);
        this.vCache[i] = f(C.maxSeq * C.kvHeads * C.headDim);
      }
    }

    this.uEmbed = uniformMixed(d, [
      ['u32', C.hidden],
      ['u32', C.hidden / 32],
      ['u32', 0],
      ['u32', 0],
    ]);
    this.uAttn = uniformU32(d, [C.heads, C.kvHeads, C.headDim, 1]);
    this.uKv = uniformU32(d, [0, C.kvHeads, C.headDim, 0]);
    this.uRms = uniformMixed(d, [
      ['u32', C.hidden],
      ['u32', 1],
      ['f32', C.rmsEps],
      ['f32', 0],
    ]);
    this.uAdd = uniformU32(d, [C.hidden, 0, 0, 0]);
    this.uConv = uniformU32(d, [C.convDim, 0, 0, 0]);
    this.uMamba = uniformMixed(d, [
      ['f32', C.dtMin],
      ['f32', 0],
      ['f32', 0],
      ['f32', 0],
    ]);
    this.uGated = uniformMixed(d, [
      ['u32', C.groupRms],
      ['u32', C.nGroups],
      ['f32', C.rmsEps],
      ['f32', 0],
    ]);
    this.uArgmax = uniformU32(d, [C.vocab, Math.ceil(C.vocab / 256), 0, 0]);
    this.gemvU = new Map();
  }

  gemvParams(K, N, nBlocks, epilogue = 0) {
    const key = `${K}:${N}:${nBlocks}:${epilogue}`;
    if (!this.gemvU.has(key)) {
      this.gemvU.set(key, uniformU32(this.device, [K, N, nBlocks, epilogue]));
    }
    return this.gemvU.get(key);
  }

  gemvPick(q4) {
    if (this.p.gemvF16 && q4.N === C.intermediate && q4.K === C.hidden) {
      return { pipeline: this.p.gemvF16, cols: 4 };
    }
    if (this.p.gemvF16 && q4.K === C.intermediate && q4.N === C.hidden) {
      return { pipeline: this.p.gemvF16, cols: 4 };
    }
    if (q4.K === C.hidden && q4.N >= 8192 && q4.N < 65536) {
      return { pipeline: this.p.gemv32, cols: 8 };
    }
    return { pipeline: this.p.gemv, cols: 4 };
  }

  gemvBg(A, q4, Y, label, epilogue = 0) {
    const { pipeline, cols } = this.gemvPick(q4);
    return {
      pipeline,
      cols,
      N: q4.N,
      bg: bg(this.device, pipeline, [
        A,
        q4.quant,
        q4.scales,
        q4.zp,
        Y,
        this.gemvParams(q4.K, q4.N, q4.nBlocks, epilogue),
      ], label),
    };
  }

  _bind() {
    const d = this.device;
    const p = this.p;
    this.layers = [];

    this.bEmbed = bg(d, p.embed, [
      this.W.get('model_embed_tokens_weight_quant'),
      this.W.get('model_embed_tokens_weight_scales'),
      this.W.get('model_embed_tokens_weight_zp'),
      this.h,
      this.tokenOut,
      this.uEmbed,
    ], 'embed');

    for (let i = 0; i < C.layers; i++) {
      const t = LAYER_TYPES[i];
      const ln = this.W.get(`model.layers.${i}.input_layernorm.weight`);
      const rmsIn = bg(d, p.rms, [this.h, ln, this.hNorm, this.uRms], `L${i}-rms`);
      const residual = bg(d, p.add, [this.h, this.mixer, this.uAdd], `L${i}-add`);
      if (t === 'mlp') {
        const up = this.q4(gemvName(i, 'mlp_up_proj'));
        const down = this.q4(gemvName(i, 'mlp_down_proj'));
        this.layers.push({
          type: t,
          rmsIn,
          up: this.gemvBg(this.hNorm, up, this.mlp, `L${i}-up`, 1),
          down: this.gemvBg(this.mlp, down, this.mixer, `L${i}-down`),
          residual,
        });
      } else if (t === 'mamba') {
        const inn = this.q4(gemvName(i, 'mamba_in_proj'));
        const out = this.q4(gemvName(i, 'mamba_out_proj'));
        const convW = this.W.get(`model.layers.${i}.mamba.conv1d.weight_squeezed`);
        const convB = this.W.get(`model.layers.${i}.mamba.conv1d.bias`);
        const aux = this.W.get(`model.layers.${i}.mamba.ssd_aux`);
        const gW = this.W.get(`model.layers.${i}.mamba.norm.weight`);
        this.layers.push({
          type: t,
          rmsIn,
          inn: this.gemvBg(this.hNorm, inn, this.mambaIn, `L${i}-in`),
          conv: bg(d, p.conv, [
            slice(this.mambaIn, 7680 * 4, C.convDim * 4),
            convW,
            convB,
            this.convState[i],
            this.convOut,
            this.uConv,
          ], `L${i}-conv`),
          ssd: bg(d, p.mamba, [
            slice(this.convOut, 0, INNER * 4),
            slice(this.convOut, 7680 * 4, 1024 * 4),
            slice(this.convOut, 8704 * 4, 1024 * 4),
            slice(this.mambaIn, 17408 * 4, 96 * 4),
            aux,
            this.ssmState[i],
            this.ssdY,
            this.uMamba,
          ], `L${i}-ssd`),
          gated: bg(d, p.gated, [
            this.ssdY,
            slice(this.mambaIn, 0, INNER * 4),
            gW,
            this.gated,
            this.uGated,
          ], `L${i}-gated`),
          out: this.gemvBg(this.gated, out, this.mixer, `L${i}-out`),
          residual,
        });
      } else if (t === 'attention') {
        const q = this.q4(gemvName(i, 'attn_q_proj'));
        const k = this.q4(gemvName(i, 'attn_k_proj'));
        const v = this.q4(gemvName(i, 'attn_v_proj'));
        const o = this.q4(gemvName(i, 'attn_o_proj'));
        this.layers.push({
          type: t,
          rmsIn,
          q: this.gemvBg(this.hNorm, q, this.q, `L${i}-q`),
          k: this.gemvBg(this.hNorm, k, this.k, `L${i}-k`),
          v: this.gemvBg(this.hNorm, v, this.v, `L${i}-v`),
          kvK: bg(d, p.kv, [this.k, this.kCache[i], this.uKv], `L${i}-kvK`),
          kvV: bg(d, p.kv, [this.v, this.vCache[i], this.uKv], `L${i}-kvV`),
          attn: bg(d, p.attn, [this.q, this.kCache[i], this.vCache[i], this.attnO, this.uAttn], `L${i}-attn`),
          o: this.gemvBg(this.attnO, o, this.mixer, `L${i}-o`),
          residual,
        });
      } else {
        const unknown = t;
        throw new Error(`unknown layer type ${unknown} at ${i}`);
      }
    }

    const finalLn = this.W.get('model.layers.42.final_norm_layernorm.weight');
    this.bFinalRms = bg(d, p.rms, [this.h, finalLn, this.hNorm, this.uRms], 'final-rms');
    const head = this.q4('lm_head_MatMul_weight');
    this.bHead = this.gemvBg(this.hNorm, head, this.logits, 'lm-head');
    this.bArgmax0 = bg(d, p.argmax0, [this.logits, this.argmaxPartial, this.uArgmax], 'argmax0');
    this.bArgmax1 = bg(d, p.argmax1, [this.tokenOut, this.argmaxPartial], 'argmax1');
    this._cachedOps = this._ops();
    this._prefillOps = this._cachedOps.filter(
      (op) => op.label !== 'final-rms' && op.label !== 'lm-head' && !op.label.startsWith('argmax'),
    );
  }

  _setU32(buf, index, value) {
    this.device.queue.writeBuffer(buf, index * 4, new Uint32Array([value]));
  }

  _gemvOp(g, label) {
    return { p: g.pipeline, bg: g.bg, wg: Math.ceil(g.N / g.cols), label };
  }

  _ops() {
    let batch = 0;
    const ops = [];
    const add = (op, newPass) => {
      if (newPass) batch += 1;
      ops.push({ ...op, batch });
    };
    add({ p: this.p.embed, bg: this.bEmbed, wg: 1, label: 'embed' }, true);
    for (let i = 0; i < this.layers.length; i++) {
      const L = this.layers[i];
      add({ p: this.p.rms, bg: L.rmsIn, wg: 1, label: `L${i}-rms` }, true);
      if (L.type === 'mlp') {
        add(this._gemvOp(L.up, `L${i}-up`), true);
        add(this._gemvOp(L.down, `L${i}-down`), true);
        add({ p: this.p.add, bg: L.residual, wg: Math.ceil(C.hidden / 256), label: `L${i}-add` }, false);
      } else if (L.type === 'mamba') {
        add(this._gemvOp(L.inn, `L${i}-in`), true);
        add({ p: this.p.conv, bg: L.conv, wg: Math.ceil(C.convDim / 256), label: `L${i}-conv` }, true);
        add({
          p: this.p.mamba,
          bg: L.ssd,
          wg: Math.ceil((C.mambaHeads * C.mambaHeadDim) / 256),
          label: `L${i}-ssd`,
        }, false);
        add({ p: this.p.gated, bg: L.gated, wg: C.nGroups, label: `L${i}-gated` }, false);
        add(this._gemvOp(L.out, `L${i}-out`), true);
        add({ p: this.p.add, bg: L.residual, wg: Math.ceil(C.hidden / 256), label: `L${i}-add` }, false);
      } else if (L.type === 'attention') {
        add(this._gemvOp(L.q, `L${i}-q`), true);
        add(this._gemvOp(L.k, `L${i}-k`), false);
        add(this._gemvOp(L.v, `L${i}-v`), false);
        add({ p: this.p.kv, bg: L.kvK, wg: Math.ceil((C.kvHeads * C.headDim) / 256), label: `L${i}-kvK` }, true);
        add({ p: this.p.kv, bg: L.kvV, wg: Math.ceil((C.kvHeads * C.headDim) / 256), label: `L${i}-kvV` }, false);
        add({ p: this.p.attn, bg: L.attn, wg: C.heads, label: `L${i}-attn` }, false);
        add(this._gemvOp(L.o, `L${i}-o`), true);
        add({ p: this.p.add, bg: L.residual, wg: Math.ceil(C.hidden / 256), label: `L${i}-add` }, false);
      } else {
        const unknown = L.type;
        throw new Error(`unknown layer type ${unknown}`);
      }
    }
    add({ p: this.p.rms, bg: this.bFinalRms, wg: 1, label: 'final-rms' }, true);
    add(this._gemvOp(this.bHead, 'lm-head'), true);
    add({ p: this.p.argmax0, bg: this.bArgmax0, wg: 256, label: 'argmax0' }, true);
    add({ p: this.p.argmax1, bg: this.bArgmax1, wg: 1, label: 'argmax1' }, false);
    return ops;
  }

  encodeToken(encoder, lmHead = true) {
    const ops = lmHead ? this._cachedOps : this._prefillOps;
    const passes = [];
    for (const op of ops) {
      const prev = passes[passes.length - 1];
      if (!prev || prev[0].batch !== op.batch) passes.push([op]);
      else prev.push(op);
    }
    const last = passes.length - 1;
    for (let p = 0; p < passes.length; p++) {
      const group = passes[p];
      const desc = { label: group[0].label };
      if (this.hasTs && (p === 0 || p === last)) {
        const tw = { querySet: this.tsSet };
        if (p === 0) tw.beginningOfPassWriteIndex = 0;
        if (p === last) tw.endOfPassWriteIndex = 1;
        desc.timestampWrites = tw;
      }
      const pass = encoder.beginComputePass(desc);
      for (const op of group) {
        pass.setPipeline(op.p);
        pass.setBindGroup(0, op.bg);
        pass.dispatchWorkgroups(op.wg);
      }
      pass.end();
    }
  }

  async _isolateBindError() {
    const d = this.device;
    for (const op of this._ops()) {
      d.pushErrorScope('validation');
      const encoder = d.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(op.p);
      pass.setBindGroup(0, op.bg);
      pass.dispatchWorkgroups(op.wg);
      pass.end();
      d.queue.submit([encoder.finish()]);
      const err = await d.popErrorScope();
      if (err) throw new Error(`${op.label}: ${err.message}`);
    }
  }

  submitStep(tokenId, { lmHead = true, streamSlot = null } = {}) {
    if (this.seq >= C.maxSeq) throw new Error(`seq ${this.seq} exceeds maxSeq ${C.maxSeq}`);
    const d = this.device;
    if (tokenId != null) this._setU32(this.tokenOut, 0, tokenId);
    this._setU32(this.uKv, 0, this.seq);
    this._setU32(this.uAttn, 3, this.seq + 1);
    const encoder = d.createCommandEncoder();
    this.encodeToken(encoder, lmHead);
    if (lmHead) {
      encoder.copyBufferToBuffer(this.tokenOut, 0, this.tokenLog, this.outCount * 4, 4);
      this.outCount += 1;
      if (streamSlot != null) {
        encoder.copyBufferToBuffer(this.tokenOut, 0, this.streamStaging[streamSlot], 0, 4);
      }
    }
    if (this.hasTs) {
      encoder.resolveQuerySet(this.tsSet, 0, 2, this.tsResolve, 0);
      encoder.copyBufferToBuffer(this.tsResolve, 0, this.tsLog, this.gpuSamples * 16, 16);
    }
    d.queue.submit([encoder.finish()]);
    this.seq += 1;
    this.gpuSamples += 1;
  }

  async step(tokenId, opts) {
    const check = this._checkErrors !== false;
    const d = this.device;
    if (check) d.pushErrorScope('validation');
    this.submitStep(tokenId, opts);
    if (check) {
      const err = await d.popErrorScope();
      if (err) {
        try {
          await this._isolateBindError();
        } catch (e2) {
          throw new Error(`${err.message} | isolated: ${e2.message}`);
        }
        throw new Error(err.message);
      }
      this._checkErrors = false;
    }
  }

  async readOutputs() {
    const n = this.outCount;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.tokenLog, 0, this.tokenStaging, 0, n * 4);
    if (this.hasTs && this.gpuSamples) {
      encoder.copyBufferToBuffer(this.tsLog, 0, this.tsStaging, 0, this.gpuSamples * 16);
    }
    this.device.queue.submit([encoder.finish()]);
    await this.tokenStaging.mapAsync(GPUMapMode.READ);
    const ids = Array.from(new Uint32Array(this.tokenStaging.getMappedRange().slice(0, n * 4)));
    this.tokenStaging.unmap();
    if (this.hasTs && this.gpuSamples) {
      await this.tsStaging.mapAsync(GPUMapMode.READ);
      const ts = new BigUint64Array(this.tsStaging.getMappedRange().slice(0, this.gpuSamples * 16));
      this.tsStaging.unmap();
      this.gpuNs = 0;
      for (let i = 0; i < this.gpuSamples; i++) {
        this.gpuNs += Number(ts[i * 2 + 1] - ts[i * 2]);
      }
    }
    return ids;
  }

  reset() {
    this.seq = 0;
    this.outCount = 0;
    this.gpuNs = 0;
    this.gpuSamples = 0;
    const z = this._zero ?? (this._zero = new Uint8Array(1 << 20));
    const wipe = (buf, bytes) => {
      for (let o = 0; o < bytes; o += z.byteLength) {
        const n = Math.min(z.byteLength, bytes - o) & ~3;
        if (n > 0) this.device.queue.writeBuffer(buf, o, z, 0, n);
      }
    };
    for (let i = 0; i < C.layers; i++) {
      if (this.convState[i]) wipe(this.convState[i], C.convDim * C.convKernel * 4);
      if (this.ssmState[i]) wipe(this.ssmState[i], C.mambaHeads * C.mambaHeadDim * C.ssmState * 4);
      if (this.kCache[i]) wipe(this.kCache[i], C.maxSeq * C.kvHeads * C.headDim * 4);
      if (this.vCache[i]) wipe(this.vCache[i], C.maxSeq * C.kvHeads * C.headDim * 4);
    }
  }

  async generate(promptIds, newTokens) {
    if (!promptIds.length) throw new Error('empty prompt');
    this.gpuNs = 0;
    this.gpuSamples = 0;
    this.outCount = 0;
    await this.step(Number(promptIds[0]), { lmHead: promptIds.length === 1 });
    for (let i = 1; i < promptIds.length; i++) {
      this.submitStep(Number(promptIds[i]), { lmHead: i === promptIds.length - 1 });
    }
    for (let i = 1; i < newTokens; i++) {
      this.submitStep(null, { lmHead: true });
    }
    return this.readOutputs();
  }
}
