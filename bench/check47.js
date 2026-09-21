import { createBuffer, createEmptyStorage, createUniform, requestGpu, adapterSummary } from '../kernels/gpu.js';
import { gemvRef, ssdRef } from '../engine47/ref.js';
import { colsPerWorkgroup, probeSubgroupSize } from '../engine47/subgroup.js';
import gemvF32Wgsl from '../kernels47/gemv_f32.wgsl?raw';
import gemvF16Wgsl from '../kernels47/gemv_f16.wgsl?raw';
import ssdWgsl from '../kernels47/mamba_ssd.wgsl?raw';

const logEl = document.getElementById('log');
const lines = [];
function say(s) {
  lines.push(s);
  logEl.textContent = lines.join('\n');
  console.log('BENCH_LOG ' + s);
}

async function postResults(payload) {
  try {
    await fetch('/__results', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch {
    /* headless still reads the console line */
  }
  console.log('BENCH_RESULT ' + JSON.stringify(payload));
}

async function shaderModule(device, code, label) {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    throw new Error(errors.map((m) => `${label} ${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
  }
  return module;
}

function pipeline(device, module, label, constants) {
  return device.createComputePipeline({
    label,
    layout: 'auto',
    compute: constants
      ? { module, entryPoint: 'main', constants }
      : { module, entryPoint: 'main' },
  });
}

function upload(device, data, usage) {
  return createBuffer(device, data, usage);
}

async function readF32(device, buffer, count) {
  const bytes = count * 4;
  const staging = device.createBuffer({
    size: Math.max(bytes, 4),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0, bytes));
  staging.unmap();
  staging.destroy();
  return out;
}

function maxAbs(a, b) {
  let m = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s;
  };
}

async function runGemv(device, pipe, colsPerWg, { quant, scales, zp, a, N, nBlocks }) {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const bQ = upload(device, quant, usage);
  const bS = upload(device, scales, usage);
  const bZ = upload(device, zp, usage);
  const bA = upload(device, a, usage);
  const bY = createEmptyStorage(device, N * 4);
  const bP = createUniform(device, new Uint32Array([N, 0, 0, 0]));
  const group = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [bA, bQ, bS, bZ, bY, bP].map((buffer, i) => ({ binding: i, resource: { buffer } })),
  });
  device.pushErrorScope('validation');
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(N / colsPerWg));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const err = await device.popErrorScope();
  if (err) throw new Error(err.message);
  const y = await readF32(device, bY, N);
  for (const b of [bQ, bS, bZ, bA, bY, bP]) b.destroy();
  return y;
}

function makeCase(N, nBlocks, seed, epilogue) {
  const rnd = lcg(seed);
  const quant = new Uint8Array(N * nBlocks * 16);
  for (let i = 0; i < quant.length; i++) quant[i] = rnd() & 0xff;
  const scales = new Float32Array(N * nBlocks);
  for (let i = 0; i < scales.length; i++) scales[i] = ((rnd() % 200) - 100) / 50;
  const zpU32 = new Uint32Array(Math.ceil((N * nBlocks) / 8));
  for (let i = 0; i < zpU32.length; i++) zpU32[i] = rnd();
  const a = new Float32Array(nBlocks * 32);
  for (let i = 0; i < a.length; i++) a[i] = ((rnd() % 100) - 50) / 25;
  return { quant, scales, zp: zpU32, a, N, nBlocks, epilogue };
}

function handCase(epilogue) {
  const quant = new Uint8Array(16).fill(0x33);
  const scales = new Float32Array([2]);
  const zp = new Uint32Array([1]);
  const a = new Float32Array(32).fill(1);
  return { quant, scales, zp, a, N: 1, nBlocks: 1, epilogue };
}

async function runSsd(device, pipe) {
  const HD = 80;
  const HEADS = 96;
  const N = 128;
  const rnd = lcg(7);
  const x = new Float32Array(HEADS * HD);
  const B = new Float32Array(8 * N);
  const C = new Float32Array(8 * N);
  const dt = new Float32Array(HEADS);
  const aux = new Float32Array(7680 + 7680 + 96);
  const H = new Float32Array(HEADS * HD * N);
  const fill = (arr, scale) => {
    for (let i = 0; i < arr.length; i++) arr[i] = ((rnd() % 100) - 50) / scale;
  };
  fill(x, 40);
  fill(B, 40);
  fill(C, 40);
  fill(dt, 20);
  fill(aux, 40);
  fill(H, 40);
  const dtMin = 0.001;
  const href = H.slice();
  const yRef = ssdRef({ x, B, C, dt, aux, H: href, dtMin });
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const bufs = [x, B, C, dt, aux, H].map((a) => upload(device, a, usage));
  const yBuf = createEmptyStorage(device, x.length * 4);
  const params = createUniform(device, new Float32Array([dtMin, 0, 0, 0]));
  const group = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [...bufs, yBuf, params].map((buffer, i) => ({ binding: i, resource: { buffer } })),
  });
  device.pushErrorScope('validation');
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil((HEADS * HD) / 256));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const err = await device.popErrorScope();
  if (err) throw new Error('ssd ' + err.message);
  const y = await readF32(device, yBuf, x.length);
  const hOut = await readF32(device, bufs[5], H.length);
  return {
    y: maxAbs(y, yRef),
    h: maxAbs(hOut, href),
  };
}

async function main() {
  const failures = [];
  const checks = [];
  try {
    const gpu = await requestGpu();
    const adapter = adapterSummary(gpu);
    const d = gpu.device;
    say('adapter ' + JSON.stringify(adapter));
    const subKeys = Object.keys(gpu.adapter.limits).filter((k) => /sub|workgroup|invocation/i.test(k));
    const limitDump = {};
    for (const k of subKeys) limitDump[k] = gpu.adapter.limits[k];
    say('limits ' + JSON.stringify(limitDump));
    say(`device subgroup ${d.limits.minSubgroupSize}..${d.limits.maxSubgroupSize} f16=${d.features.has('shader-f16')}`);
    const subgroupSize = await probeSubgroupSize(d);
    const colsPerWg = colsPerWorkgroup(subgroupSize);
    say(`probed subgroup ${subgroupSize} cols/wg ${colsPerWg}`);
    const f32mod = await shaderModule(d, gemvF32Wgsl, 'gemv-f32');
    say('compiled gemv f32');
    let f16mod = null;
    if (d.features.has('shader-f16')) {
      f16mod = await shaderModule(d, gemvF16Wgsl, 'gemv-f16');
      say('compiled gemv f16');
    }
    const ssdMod = await shaderModule(d, ssdWgsl, 'ssd');
    const ssdPipe = pipeline(d, ssdMod, 'ssd');
    say('compiled ssd');
    if (!subgroupSize) {
      say('skip numeric GEMV: subgroup probe returned 0');
    } else {
      const specs = [
        ['hand', handCase(0), 0],
        ['hand-relu2', handCase(1), 0],
        ['tail-33', makeCase(65, 33, 3, 0), 1e-3],
        ['k3136', makeCase(64, 98, 9, 0), 1e-3],
      ];
      for (const [name, spec, tol] of specs) {
        const pipe = pipeline(d, f32mod, name, { N_BLOCKS: spec.nBlocks, EPILOGUE: spec.epilogue });
        const got = await runGemv(d, pipe, colsPerWg, spec);
        const exp = gemvRef(spec.quant, spec.scales, spec.zp, spec.a, spec.N, spec.nBlocks, spec.epilogue);
        const err = maxAbs(got, exp);
        let mismatch = '';
        for (let i = 0; i < exp.length; i++) {
          if (Math.abs(got[i] - exp[i]) > 1e-3) {
            mismatch = ` [${i}] got=${got[i]} exp=${exp[i]}`;
            break;
          }
        }
        checks.push({ name, err, tol });
        say(`${name} maxAbs=${err}${mismatch}`);
        if (err > tol) failures.push(`${name} maxAbs ${err} > ${tol}`);
      }
      if (f16mod) {
        const spec = handCase(1);
        const pipe = pipeline(d, f16mod, 'f16-hand', { N_BLOCKS: 1, EPILOGUE: 1 });
        const got = await runGemv(d, pipe, colsPerWg, spec);
        const err = Math.abs(got[0] - 16384);
        checks.push({ name: 'f16-hand-relu2', err, tol: 0 });
        say(`f16-hand-relu2 maxAbs=${err} got=${got[0]}`);
        if (err > 0) failures.push(`f16 hand relu2 got ${got[0]}`);
      }
    }
    const ssdErr = await runSsd(d, ssdPipe);
    checks.push({ name: 'ssd-y', err: ssdErr.y, tol: 1e-3 });
    checks.push({ name: 'ssd-h', err: ssdErr.h, tol: 1e-3 });
    say(`ssd y maxAbs=${ssdErr.y} h maxAbs=${ssdErr.h}`);
    if (ssdErr.y > 1e-3 || ssdErr.h > 1e-3) failures.push(`ssd y=${ssdErr.y} h=${ssdErr.h}`);
    const payload = {
      type: 'check47',
      ok: failures.length === 0,
      adapter,
      subgroup: [d.limits.minSubgroupSize, d.limits.maxSubgroupSize],
      checks,
      failures,
    };
    say(payload.ok ? 'PASS' : 'FAIL ' + failures.join('; '));
    await postResults(payload);
  } catch (e) {
    const msg = e && e.stack ? e.stack : String(e);
    say('BENCH_ERROR ' + msg);
    await postResults({ type: 'check47', ok: false, error: msg, failures });
  }
}

main();
