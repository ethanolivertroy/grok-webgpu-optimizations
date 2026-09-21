import test from 'node:test';
import assert from 'node:assert/strict';
import { dequantDot, gemvRef, loadZp } from './ref.js';

test('zp nibble 0 is the low 4 bits', () => {
  const zp = new Uint32Array([0x1]);
  assert.equal(loadZp(zp, 0, 1, 0), 1);
});

test('one block of nibble 3, zp 1, scale 2, A = 1 is 128', () => {
  const nBlocks = 1;
  const quant = new Uint8Array(16).fill(0x33);
  const scales = new Float32Array([2]);
  const zp = new Uint32Array([1]);
  const a = new Float32Array(32).fill(1);
  assert.equal(dequantDot(quant, scales, zp, 0, nBlocks, a), 128);
  const y = gemvRef(quant, scales, zp, a, 1, nBlocks, 1);
  assert.equal(y[0], 16384);
});

test('column 1 uses its own scale', () => {
  const nBlocks = 1;
  const quant = new Uint8Array(32).fill(0x11);
  const scales = new Float32Array([2, 4]);
  const zp = new Uint32Array([0]);
  const a = new Float32Array(32).fill(1);
  // nibble 1, zp 0, 32 weights: col0 = 32 * 1 * 2, col1 = 32 * 1 * 4
  const y = gemvRef(quant, scales, zp, a, 2, nBlocks, 0);
  assert.equal(y[0], 64);
  assert.equal(y[1], 128);
});
