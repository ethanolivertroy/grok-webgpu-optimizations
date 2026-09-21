import subgroupProbeWgsl from '../kernels47/subgroup_probe.wgsl?raw';
import { createEmptyStorage } from '../kernels/gpu.js';

export const GEMV_WG = 256;
export const GEMV_COLS = 8;

export async function probeSubgroupSize(device) {
  const module = device.createShaderModule({ label: 'subgroup-probe', code: subgroupProbeWgsl });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    throw new Error(errors.map((m) => m.message).join('\n'));
  }
  const pipeline = device.createComputePipeline({
    label: 'subgroup-probe',
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const out = createEmptyStorage(device, 16);
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: out } }],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(1);
  pass.end();
  const staging = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  encoder.copyBufferToBuffer(out, 0, staging, 0, 16);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(staging.getMappedRange().slice(0, 16));
  staging.unmap();
  staging.destroy();
  out.destroy();
  const size = words[0];
  if (!size || GEMV_WG % size !== 0) {
    throw new Error(`subgroup size ${size} does not divide workgroup ${GEMV_WG}`);
  }
  return size;
}

export function colsPerWorkgroup(subgroupSize) {
  return (GEMV_WG / subgroupSize) * GEMV_COLS;
}
