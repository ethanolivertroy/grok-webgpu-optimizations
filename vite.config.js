import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));

function copyVideo() {
  return {
    name: 'copy-video',
    configureServer(server) {
      server.middlewares.use('/video', (req, res, next) => {
        const name = decodeURIComponent((req.url || '').split('?')[0].replace(/^\//, ''));
        if (!/^[A-Za-z0-9._-]+$/.test(name)) return next();
        const file = path.join(root, 'video', name);
        if (!fs.existsSync(file)) return next();
        res.setHeader('content-type', 'video/mp4');
        fs.createReadStream(file).pipe(res);
      });
    },
    writeBundle() {
      if (process.env.SPACE_BUILD === '1') {
        fs.copyFileSync(path.join(root, 'space', 'README.md'), path.join(root, 'dist', 'README.md'));
        return;
      }
      const destDir = path.join(root, 'dist', 'video');
      fs.mkdirSync(destDir, { recursive: true });
      const src = path.join(root, 'video', 'nemotron-decode.mp4');
      if (fs.existsSync(src)) {
        fs.copyFileSync(src, path.join(destDir, 'nemotron-decode.mp4'));
      }
    },
  };
}

function weightsStatic() {
  const dir = path.join(root, 'harness/weights');
  return {
    name: 'weights-static',
    configureServer(server) {
      server.middlewares.use('/weights', (req, res, next) => {
        const name = decodeURIComponent((req.url || '').split('?')[0].replace(/^\//, ''));
        if (!/^[A-Za-z0-9._-]+$/.test(name)) return next();
        const file = path.join(dir, name);
        if (!fs.existsSync(file)) {
          res.statusCode = 404;
          res.end(`missing ${name}; put ONNX data files in harness/weights/`);
          return;
        }
        const stat = fs.statSync(file);
        const range = req.headers.range;
        res.setHeader('content-type', 'application/octet-stream');
        res.setHeader('accept-ranges', 'bytes');
        if (range) {
          const m = range.match(/bytes=(\d+)-(\d*)/);
          const start = Number(m[1]);
          const end = m[2] ? Number(m[2]) : stat.size - 1;
          res.statusCode = 206;
          res.setHeader('content-range', `bytes ${start}-${end}/${stat.size}`);
          res.setHeader('content-length', String(end - start + 1));
          fs.createReadStream(file, { start, end }).pipe(res);
          return;
        }
        res.setHeader('content-length', String(stat.size));
        fs.createReadStream(file).pipe(res);
      });
    },
  };
}

function resultsCollector() {
  return {
    name: 'results-collector',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url !== '/__results' || req.method !== 'POST') return next();
        const chunks = [];
        req.on('data', (c) => chunks.push(c));
        req.on('end', () => {
          const body = Buffer.concat(chunks).toString('utf8');
          const out = path.resolve('harness/last-run.json');
          fs.mkdirSync(path.dirname(out), { recursive: true });
          fs.writeFileSync(out, body);
          res.statusCode = 200;
          res.setHeader('content-type', 'text/plain');
          res.end('ok');
        });
      });
    },
  };
}

const space = process.env.SPACE_BUILD === '1';

export default defineConfig({
  plugins: [weightsStatic(), resultsCollector(), copyVideo()],
  base: './',
  publicDir: space ? false : 'public',
  server: {
    port: 5173,
    host: '127.0.0.1',
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    port: 4173,
    host: '127.0.0.1',
  },
  worker: { format: 'es' },
  build: {
    target: 'esnext',
    rollupOptions: {
      input: space
        ? { main: path.join(root, 'index.html') }
        : {
            main: path.join(root, 'index.html'),
            charts: path.join(root, 'charts/chart.html'),
            engine: path.join(root, 'bench/engine.html'),
            decode: path.join(root, 'bench/decode.html'),
            compare: path.join(root, 'bench/compare.html'),
            check47: path.join(root, 'bench/check47.html'),
          },
    },
  },
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
});
