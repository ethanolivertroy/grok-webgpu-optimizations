#!/usr/bin/env node
/**
 * Headless bench driver: starts Vite, launches full Chromium, waits for the
 * bench page to POST /__results (written to harness/last-run.json), captures
 * the JSON per run, prints the combined result.
 *
 * Usage:
 *   node scripts/headless-bench.mjs --page bench --angle vulkan [--iters 50]
 *   node scripts/headless-bench.mjs --page auto [--mode stock|both] [--tokens 128]
 *
 * --angle vulkan uses the real GPU (NVIDIA ICD); swiftshader = CPU emulation.
 * auto mode: 'both' runs stock then custom with a shared profile dir.
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const resultsPath = path.join(root, 'harness', 'last-run.json');
const CHROME =
  process.env.CHROME_PATH ||
  (() => {
    const candidates = fs
      .readdirSync('/root/.cache/ms-playwright', { withFileTypes: true })
      .filter((d) => d.isDirectory() && d.name.startsWith('chromium-'))
      .map((d) => path.join('/root/.cache/ms-playwright', d.name, 'chrome-linux64', 'chrome'))
      .filter(fs.existsSync);
    return candidates.sort().at(-1);
  })();
const PORT = process.env.PORT || '5199';
const PAGE_TIMEOUT_MS = parseInt(process.env.PAGE_TIMEOUT_MS || (20 * 60 * 1000), 10);

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => {
    const i = a.indexOf(k);
    return i >= 0 ? a[i + 1] : undefined;
  };
  return {
    page: get('--page') || 'bench',
    angle: get('--angle') || 'swiftshader',
    mode: get('--mode') || 'both',
    iters: get('--iters') || '50',
    tokens: get('--tokens') || '128',
    profileDir: get('--profile-dir'),
    headed: a.includes('--headed'),
  };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// macOS VM quirk: a SIGKILLed chrome leaves Singleton*/LevelDB LOCK files that
// hang the NEXT chrome session's startup (page never executes). Sweep them.
function clearProfileLocks(profileDir) {
  try {
    for (const f of fs.readdirSync(profileDir)) {
      if (f.startsWith('Singleton')) fs.rmSync(path.join(profileDir, f), { force: true });
    }
    const walk = (dir) => {
      for (const f of fs.readdirSync(dir)) {
        const p = path.join(dir, f);
        try {
          if (f === 'LOCK') fs.rmSync(p, { force: true });
          else if (fs.statSync(p).isDirectory()) walk(p);
        } catch { /* gone */ }
      }
    };
    walk(profileDir);
  } catch { /* best effort */ }
}

async function runChromePage(url, profileDir, angle, out, headed) {
  clearProfileLocks(profileDir);
  const args = [
    ...(headed ? [] : ['--headless=new']),
    '--no-sandbox',
    '--disable-gpu-sandbox',
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
    `--user-data-dir=${profileDir}`,
  ];
  if (angle === 'vulkan') {
    args.push('--use-angle=vulkan', '--enable-features=Vulkan');
  } else if (angle === 'metal') {
    args.push('--use-angle=metal');
  } else args.push('--use-angle=swiftshader');
  // Surface page console + errors on stderr (macOS needs this for visibility)
  args.push('--enable-logging=stderr');
  const extra = process.env.CHROME_EXTRA_ARGS;
  if (extra) args.push(...extra.split(' ').filter(Boolean));
  args.push(url);

  if (fs.existsSync(resultsPath)) fs.unlinkSync(resultsPath);

  const chrome = spawn(CHROME, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  chrome.stdout.on('data', () => {});
  chrome.stderr.on('data', (d) => {
    const s = d.toString();
    for (const line of s.split('\n')) {
      if (line.includes('BENCH_ERROR')) out.errors.push(line.trim());
      if (line.includes('BENCH_RESULT ')) {
        // console channel: page reports via console when no /__results POST exists
        try {
          const payload = line.slice(line.indexOf('BENCH_RESULT ') + 13).trim();
          out.consoleResults.push(JSON.parse(payload));
        } catch { /* malformed */ }
      }
      if (/CONSOLE|vulkan|dawn|gpu/i.test(line)) {
        if (out.chromeWarnings.length < 60) out.chromeWarnings.push(line.trim());
        // stream page console live so a hung run shows where it stuck
        if (line.includes('BENCH_LOG') || line.includes('BENCH_ERROR') || line.includes('WATCHDOG')) {
          process.stderr.write(line.slice(0, 400) + '\n');
        }
      }
    }
  });

  const deadline = Date.now() + PAGE_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await sleep(500);
    // console channel (static serving / no POST endpoint)
    if (out.consoleResults.length) {
      chrome.kill();
      await waitChromeExit(chrome, profileDir);
      return out.consoleResults.shift();
    }
    if (fs.existsSync(resultsPath)) {
      try {
        const json = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
        if (json && (json.rows || json.type)) {
          chrome.kill();
          await waitChromeExit(chrome, profileDir);
          return json;
        }
      } catch {
        /* partial write, keep waiting */
      }
    }
  }
  chrome.kill();
  await waitChromeExit(chrome, profileDir);
  return null;
}

async function waitChromeExit(chrome, profileDir) {
  if (chrome.exitCode === null) {
    await Promise.race([
      new Promise((r) => chrome.once('exit', r)),
      sleep(5000),
    ]);
  }
  // macOS can leave helper processes holding the profile lock; sweep them.
  // Match --user-data-dir (chrome's arg) NOT --profile-dir (our own arg),
  // otherwise pkill -f matches this driver's own command line and kills us.
  try {
    spawn('pkill', ['-f', `user-data-dir=${profileDir}`]);
  } catch {
    /* pkill missing */
  }
  await sleep(800);
}

async function main() {
  const cfg = parseArgs();
  if (!CHROME || !fs.existsSync(CHROME)) {
    console.error('chromium not found; set CHROME_PATH');
    process.exit(1);
  }
  console.error(`chrome: ${CHROME} (angle=${cfg.angle})`);

  const profileDir =
    cfg.profileDir || fs.mkdtempSync(path.join('/tmp', 'chrome-bench-'));
  const out = { results: [], errors: [], chromeWarnings: [], consoleResults: [] };
  const base = `http://127.0.0.1:${PORT}`;

  if (process.env.NO_VITE !== '1') {
    const vite = spawn('npx', ['vite', '--host', '127.0.0.1', '--port', PORT, '--strictPort'], {
      cwd: root,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, BROWSER: 'none' },
    });
    let ready = false;
    vite.stdout.on('data', (d) => {
      const s = d.toString();
      if (s.includes('Local:') || s.includes('localhost')) ready = true;
    });
    vite.stderr.on('data', (d) => process.stderr.write(d));
    const t0 = Date.now();
    while (!ready && Date.now() - t0 < 30000) await sleep(200);
    if (!ready) {
      vite.kill();
      console.error('vite did not start');
      process.exit(1);
    }
    globalThis.__vite = vite;
  }

  if (cfg.page === 'bench') {
    const json = await runChromePage(
      `${base}/bench/gpu-bench.html?iters=${cfg.iters}`,
      profileDir,
      cfg.angle,
      out,
      cfg.headed,
    );
    if (json) out.results.push(json);
  } else if (cfg.page === 'engine') {
    const json = await runChromePage(
      `${base}/bench/engine.html?iters=${cfg.iters}`,
      profileDir,
      cfg.angle,
      out,
      cfg.headed,
    );
    if (json) out.results.push(json);
  } else if (cfg.page === 'decode') {
    const json = await runChromePage(
      `${base}/bench/decode.html?tokens=${cfg.tokens}`,
      profileDir,
      cfg.angle,
      out,
      cfg.headed,
    );
    if (json) out.results.push(json);
  } else if (cfg.page === 'check') {
    const json = await runChromePage(
      `${base}/bench/check47.html`,
      profileDir,
      cfg.angle,
      out,
      cfg.headed,
    );
    if (json) out.results.push(json);
  } else if (cfg.page === 'compare') {
    const json = await runChromePage(
      `${base}/bench/compare.html?tokens=${cfg.tokens}`,
      profileDir,
      cfg.angle,
      out,
      cfg.headed,
    );
    if (json) out.results.push(json);
  } else if (cfg.page === 'ort-first') {
    const json = await runChromePage(
      `${base}/bench/ort-first.html`,
      profileDir,
      cfg.angle,
      out,
      cfg.headed,
    );
    if (json) out.results.push(json);
  } else if (cfg.page === 'auto') {
    const modes = cfg.mode === 'both' ? ['stock', 'custom'] : [cfg.mode];
    for (const mode of modes) {
      // Chrome writes Dawn pipeline caches into the profile; SIGKILLed runs can
      // corrupt them and hang the NEXT chrome session. Clear between modes.
      for (const c of ['Default/DawnWebGPUCache', 'Default/GPUCache', 'Default/ShaderCache', 'Default/GrShaderCache']) {
        fs.rmSync(path.join(profileDir, c), { recursive: true, force: true });
      }
      const json = await runChromePage(
        `${base}/bench/auto.html?mode=${mode}&tokens=${cfg.tokens}${process.env.CACHE_OFF === '1' ? '&cache=off' : ''}${process.env.DUMP === '1' ? '&dump=1' : ''}`,
        profileDir,
        cfg.angle,
        out,
        cfg.headed,
      );
      if (json) out.results.push(json);
      if (!json) out.errors.push(`no result for mode=${mode}`);
    }
  }

  if (globalThis.__vite) globalThis.__vite.kill();
  await sleep(2000); // let chrome release the profile dir
  if (!cfg.profileDir) {
    try {
      fs.rmSync(profileDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 });
    } catch {
      /* best-effort cleanup */
    }
  }

  if (out.errors.length) console.error('BENCH ERRORS:', out.errors);
  console.log(JSON.stringify({ page: cfg.page, angle: cfg.angle, results: out.results, errors: out.errors, chromeWarnings: out.chromeWarnings }, null, 2));
  process.exit(out.results.length > 0 ? 0 : 1);
}

main();
