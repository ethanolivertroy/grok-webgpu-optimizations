/**
 * Shared WebGPU helpers for kernel numerics tests and the bench spy.
 * All measurements on this host are SwiftShader (software). Do not treat
 * timings as performance numbers.
 */

export const GPU_BUFFER = {
  STORAGE: 0x80,
  UNIFORM: 0x40,
  COPY_SRC: 0x04,
  COPY_DST: 0x08,
  MAP_READ: 0x01,
};

export async function requestGpu({ forceFallback = false } = {}) {
  if (!navigator.gpu) {
    throw new Error('WebGPU not available');
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
    forceFallbackAdapter: forceFallback,
  });
  if (!adapter) {
    throw new Error('No WebGPU adapter');
  }
  const info = adapter.info ?? {};
  const features = [...adapter.features];
  const hasF16 = adapter.features.has('shader-f16');
  const requiredFeatures = [];
  if (hasF16) requiredFeatures.push('shader-f16');
  if (adapter.features.has('timestamp-query')) requiredFeatures.push('timestamp-query');
  if (adapter.features.has('subgroups')) requiredFeatures.push('subgroups');
  if (adapter.features.has('chromium-experimental-subgroup-matrix')) {
    requiredFeatures.push('chromium-experimental-subgroup-matrix');
  }
  const requiredLimits = {
    maxComputeWorkgroupSizeX: Math.min(256, adapter.limits.maxComputeWorkgroupSizeX),
    maxComputeInvocationsPerWorkgroup: Math.min(
      256,
      adapter.limits.maxComputeInvocationsPerWorkgroup,
    ),
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBuffersPerShaderStage: Math.min(
      10,
      adapter.limits.maxStorageBuffersPerShaderStage,
    ),
  };
  // Pin simd width at 32 when the adapter allows it. Both GEMV kernels
  // (4.6 shuffle-xor and 4.7 subgroupAdd) assume a 32-lane subgroup.
  // Requesting min and max equal asks the device to fix the size.
  const pinned = { ...requiredLimits };
  if (adapter.features.has('subgroups')) {
    const mn = adapter.limits.minSubgroupSize;
    const mx = adapter.limits.maxSubgroupSize;
    if (mn <= 32 && mx >= 32) {
      pinned.minSubgroupSize = 32;
      pinned.maxSubgroupSize = 32;
    }
  }
  let device;
  try {
    device = await adapter.requestDevice({ requiredFeatures, requiredLimits: pinned });
  } catch (err) {
    if (pinned.minSubgroupSize == null) throw err;
    device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  }
  device.addEventListener('uncapturederror', (e) => {
    console.error('GPU uncaptured error', e.error);
  });
  return {
    adapter,
    device,
    hasF16,
    features,
    info: {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      device: info.device ?? '',
      description: info.description ?? '',
      isFallbackAdapter: adapter.isFallbackAdapter ?? null,
      envLabel: `${info.vendor || 'unknown'} / ${info.architecture || 'unknown'}`,
    },
  };
}

export function alignBytes(n, align = 256) {
  return Math.ceil(n / align) * align;
}

export function createBuffer(device, data, usage) {
  const bytes =
    data instanceof ArrayBuffer
      ? new Uint8Array(data)
      : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const size = alignBytes(bytes.byteLength, 4);
  const buf = device.createBuffer({
    size: Math.max(size, 4),
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(bytes);
  buf.unmap();
  return buf;
}

export function createEmptyStorage(device, byteLength, extraUsage = 0) {
  return device.createBuffer({
    size: alignBytes(Math.max(byteLength, 4), 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extraUsage,
  });
}

export function createUniform(device, values /* number[] or Uint32Array/Float32Array */) {
  let packed;
  if (values instanceof ArrayBuffer || ArrayBuffer.isView(values)) {
    packed = values instanceof Uint8Array ? values : new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  } else {
    const u32 = new Uint32Array(values.length);
    const f32 = new Float32Array(u32.buffer);
    for (let i = 0; i < values.length; i++) {
      // Heuristic: integer-looking uniforms stay u32, else f32. Callers should
      // pass a typed array when mixed.
      u32[i] = values[i];
    }
    packed = new Uint8Array(u32.buffer);
    void f32;
  }
  const size = alignBytes(packed.byteLength, 16);
  const buf = device.createBuffer({
    size: Math.max(size, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(packed);
  buf.unmap();
  return buf;
}

export async function compileCompute(device, code, label, entryPoint = 'main') {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  const warnings = info.messages.filter((m) => m.type === 'warning');
  if (errors.length) {
    const text = errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n');
    throw new Error(`WGSL compile failed (${label}):\n${text}`);
  }
  const pipeline = device.createComputePipeline({
    label,
    layout: 'auto',
    compute: { module, entryPoint },
  });
  return { pipeline, warnings: warnings.map((m) => m.message), module };
}

export async function dispatchAndRead(device, { pipeline, bindings, dispatch, readBuffer, readBytes, readType = 'f32' }) {
  const entries = bindings.map((buffer, i) => ({
    binding: i,
    resource: { buffer },
  }));
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(dispatch[0], dispatch[1] ?? 1, dispatch[2] ?? 1);
  pass.end();
  const staging = device.createBuffer({
    size: alignBytes(readBytes, 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(readBuffer, 0, staging, 0, readBytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0, readBytes);
  staging.unmap();
  staging.destroy();
  if (readType === 'f32') return new Float32Array(copy);
  if (readType === 'u32') return new Uint32Array(copy);
  return new Uint8Array(copy);
}

export function adapterSummary(gpu) {
  const arch = gpu.info.architecture || '';
  const vendor = gpu.info.vendor || '';
  const desc = `${vendor} ${arch} ${gpu.info.description || ''}`.toLowerCase();
  const isMetal = /metal/.test(arch) || vendor === 'apple';
  const isSwift = /swiftshader|software/.test(desc);
  let env;
  if (isMetal && !isSwift) env = 'Apple Metal (real GPU)';
  else if (isSwift) env = 'SwiftShader/cloud (software WebGPU). Not a performance number.';
  else env = gpu.info.envLabel || `${vendor} / ${arch}`;
  return {
    env,
    vendor: gpu.info.vendor,
    architecture: gpu.info.architecture,
    device: gpu.info.device,
    shaderF16: gpu.hasF16,
    features: gpu.features,
  };
}
