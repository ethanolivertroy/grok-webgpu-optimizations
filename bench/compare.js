import { AutoTokenizer, env } from '@huggingface/transformers';
import { requestGpu, adapterSummary } from '../kernels/gpu.js';
import { loadWeights } from '../engine/weights.js';
import { DecodeEngine } from '../engine/runtime.js';
import { DecodeEngine47 } from '../engine47/runtime.js';

const MODEL = 'onnx-community/NVIDIA-Nemotron-3-Nano-4B-BF16-ONNX';
const PROMPT = 'Write a haiku about GPU kernels.';
const PUBLISHED = {
  stock: 69.76,
  wall46: 115.04,
  gpu46: 142.48,
  ms46: 1112.7,
  gpuMs46: 7.019,
};
const params = new URLSearchParams(location.search);
const MAX_NEW = parseInt(params.get('tokens') || '128', 10);

env.allowRemoteModels = true;
env.useBrowserCache = params.get('cache') !== 'off';

const logEl = document.getElementById('log');
const lines = [];
function say(s) {
  lines.push(s);
  logEl.textContent = lines.join('\n');
  console.log('BENCH_LOG ' + s);
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function paintBars(rows) {
  const host = document.getElementById('bars');
  const max = Math.max(...rows.map((r) => r.value), 1);
  host.replaceChildren();
  for (const row of rows) {
    const wrap = document.createElement('div');
    wrap.className = 'bar-row';
    const label = document.createElement('span');
    label.textContent = row.label;
    const track = document.createElement('div');
    track.className = 'track';
    const fill = document.createElement('i');
    fill.style.width = `${(row.value / max) * 100}%`;
    fill.style.background = row.color;
    track.appendChild(fill);
    const num = document.createElement('b');
    num.textContent = row.value == null ? '' : row.value.toFixed(1);
    wrap.append(label, track, num);
    host.appendChild(wrap);
  }
}

async function postResults(payload) {
  try {
    await fetch('/__results', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch {
    /* console line is the fallback */
  }
  console.log('BENCH_RESULT ' + JSON.stringify(payload));
}

function agree(a, b) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i += 1;
  return { n: i, same: i === a.length && i === b.length && a.length === b.length };
}

async function timeEngine(engine, promptIds, newTokens) {
  engine.reset();
  await engine.generate(promptIds, 8);
  engine.reset();
  const t0 = performance.now();
  const outIds = await engine.generate(promptIds, newTokens);
  const ms = performance.now() - t0;
  const tokPerSec = outIds.length / (ms / 1000);
  const gpuMs = engine.gpuSamples ? engine.gpuNs / 1e6 : null;
  const gpuMsPerTok = gpuMs != null && engine.gpuSamples ? gpuMs / engine.gpuSamples : null;
  const gpuTokPerSec = gpuMsPerTok ? 1000 / gpuMsPerTok : null;
  return {
    ids: outIds,
    tokens: outIds.length,
    ms: +ms.toFixed(1),
    tokPerSec: +tokPerSec.toFixed(2),
    gpu_ms: gpuMs != null ? +gpuMs.toFixed(1) : null,
    gpu_samples: engine.gpuSamples,
    gpu_ms_per_tok: gpuMsPerTok != null ? +gpuMsPerTok.toFixed(3) : null,
    gpu_tokPerSec: gpuTokPerSec != null ? +gpuTokPerSec.toFixed(2) : null,
  };
}

function fillCard(prefix, run, text) {
  setText(`${prefix}-wall`, run ? String(run.tokPerSec) : 'n/a');
  setText(`${prefix}-gpu`, run && run.gpu_tokPerSec != null ? String(run.gpu_tokPerSec) : 'n/a');
  setText(`${prefix}-ms`, run ? `${run.ms} ms` : '');
  setText(`${prefix}-out`, text || '');
}

async function main() {
  paintBars([
    { label: 'stock ORT', value: PUBLISHED.stock, color: '#5c6b7a' },
    { label: '4.6 wall, M5 Max', value: PUBLISHED.wall46, color: '#f54e00' },
    { label: '4.6 GPU, M5 Max', value: PUBLISHED.gpu46, color: '#005288' },
  ]);
  const payload = {
    type: 'decode-compare',
    published: PUBLISHED,
    tokens_requested: MAX_NEW,
    prompt: PROMPT,
  };
  try {
    say('requesting gpu');
    const gpu = await requestGpu();
    payload.adapter = adapterSummary(gpu);
    say('adapter ' + JSON.stringify(payload.adapter));
    setText('machine', payload.adapter.env);

    say('loading tokenizer');
    const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
    const templated = tokenizer.apply_chat_template(
      [{ role: 'user', content: PROMPT }],
      { add_generation_prompt: true, tokenize: false },
    );
    const promptIds = tokenizer.encode(templated, { add_special_tokens: false });
    payload.prompt_tokens = promptIds.length;
    say(`prompt tokens=${promptIds.length}`);

    say('loading q4 weights');
    let lastPct = '';
    const weights = await loadWeights(gpu.device, {
      onProgress: (name, got, total) => {
        if (!total) return;
        const pct = Math.floor((got / total) * 10) * 10;
        const key = `${name}:${pct}`;
        if (pct === 0 || key === lastPct) return;
        lastPct = key;
        say(`download ${name} ${pct}%`);
        setText('status', `download ${name} ${pct}%`);
      },
    });

    say('compiling 4.6');
    setText('status', 'compiling 4.6');
    const e46 = await DecodeEngine.fromWeights(gpu, weights);
    say('compiling 4.7');
    setText('status', 'compiling 4.7');
    let e47 = null;
    let err47 = null;
    try {
      e47 = await DecodeEngine47.create(gpu, weights);
      say(`4.7 subgroup ${e47.subgroupSize} cols/wg ${e47.colsPerWg}`);
    } catch (e) {
      err47 = e && e.message ? e.message : String(e);
      say('4.7 compile failed: ' + err47);
      setText('c47-out', err47);
    }

    say(`timed 4.6, ${MAX_NEW} tokens`);
    setText('status', 'running 4.6');
    const r46 = await timeEngine(e46, promptIds, MAX_NEW);
    const text46 = tokenizer.decode(r46.ids, { skip_special_tokens: true });
    fillCard('c46', r46, text46.slice(0, 280));
    payload.grok46 = { ...r46, output: text46.slice(0, 500) };
    delete payload.grok46.ids;
    say(`4.6 wall=${r46.tokPerSec} gpu=${r46.gpu_tokPerSec}`);

    let r47 = null;
    let text47 = '';
    if (e47) {
      say(`timed 4.7, ${MAX_NEW} tokens`);
      setText('status', 'running 4.7');
      r47 = await timeEngine(e47, promptIds, MAX_NEW);
      text47 = tokenizer.decode(r47.ids, { skip_special_tokens: true });
      fillCard('c47', r47, text47.slice(0, 280));
      payload.grok47 = { ...r47, output: text47.slice(0, 500) };
      delete payload.grok47.ids;
      const match = agree(r46.ids, r47.ids);
      payload.token_agree = match.n;
      payload.token_match = match.same;
      setText('agree', match.same
        ? `Token ids match (${match.n})`
        : `Token ids match for the first ${match.n}`);
      say(`4.7 wall=${r47.tokPerSec} gpu=${r47.gpu_tokPerSec} agree=${match.n}`);
    } else {
      payload.grok47 = { error: err47 };
    }

    const rows = [
      { label: 'stock ORT, M5', value: PUBLISHED.stock, color: '#5c6b7a' },
      { label: '4.6 wall, M5', value: PUBLISHED.wall46, color: '#f54e00' },
      { label: '4.6 GPU, M5', value: PUBLISHED.gpu46, color: '#005288' },
      { label: '4.6 wall, here', value: r46.tokPerSec, color: '#f54e00' },
    ];
    if (r46.gpu_tokPerSec != null) {
      rows.push({ label: '4.6 GPU, here', value: r46.gpu_tokPerSec, color: '#7eb6d9' });
    }
    if (r47) {
      rows.push({ label: '4.7 wall, here', value: r47.tokPerSec, color: '#c8f7c5' });
      if (r47.gpu_tokPerSec != null) {
        rows.push({ label: '4.7 GPU, here', value: r47.gpu_tokPerSec, color: '#8f8' });
      }
    }
    paintBars(rows);
    setText('status', 'done');
    say('done');
    await postResults(payload);
  } catch (e) {
    const msg = e && e.stack ? e.stack : String(e);
    say('BENCH_ERROR ' + msg);
    setText('status', 'failed');
    payload.error = msg;
    await postResults(payload);
  }
}

main();
