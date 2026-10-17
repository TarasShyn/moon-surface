// WebGPU lunar surface: six tiled procedural height layers on a CDLOD mesh, ray-marched sun
// shadows over the curved surface, an airless sky with the Earth, and HDR post.
import { TEX, LAYERS, TEX_MIPS, LAYER_P, LAYER_ROT, LAYER_OFF, MOON_R, SH_RES, SH_LEVELS } from './wgsl/common.js';
import { GEN, DERIV, HEIGHT_MIPS, SCALAR_MIPS, MAT_MIPS } from './wgsl/gen.js';
import { TERRAIN, PROBE, SHADOW_MAP } from './wgsl/terrain.js';
import { SKY } from './wgsl/sky.js';
import { BLOOM_DOWN, BLOOM_UP, COMPOSITE } from './wgsl/post.js';

const qs = new URLSearchParams(location.search);
const DBG = Number(qs.get('dbg')) || 0;
const D2R = Math.PI / 180;
const GU = GPUBufferUsage;
const TU = GPUTextureUsage;
const EARTH_ANG_R = 0.0165;
const EYE_HEIGHT = 1.7;
const LUNAR_G = 1.62;
const SH_SIZE = [64, 512, 4096, 32768, 262144];
const SH_ROWS = 128;
const BLOOM_CLAMP = 2.0;
const SYNODIC_MONTH_S = 29.530589 * 86400;
const SIDEREAL_DAY_S = 86164.1;
const CLEARANCE = 1.0;
const POLE = (() => {
  const e = 32 * Math.PI / 180;
  return [0, Math.sin(e), -Math.cos(e)];
})();

function rotateAbout(v, a, ang) {
  const c = Math.cos(ang), s = Math.sin(ang), d = (a[0] * v[0] + a[1] * v[1] + a[2] * v[2]) * (1 - c);
  return [
    v[0] * c + (a[1] * v[2] - a[2] * v[1]) * s + a[0] * d,
    v[1] * c + (a[2] * v[0] - a[0] * v[2]) * s + a[1] * d,
    v[2] * c + (a[0] * v[1] - a[1] * v[0]) * s + a[2] * d,
  ];
}

export const P = {
  sunElev: 14,
  sunAzim: 150,
  earthElev: 22,
  earthAzim: -88,
  earthBright: 1.0,
  exposure: 4.8,
  timeLapse: 4000,
  sunMoves: 0,
  albedo: 0.12,
  bounce: 1.0,
  earthshine: 1.0,
  stars: 1.0,
  milkyWay: 0.6,
  shadowSteps: 14,
  bloom: 0.06,
  fov: 58,
  scale: 1,
};

function fail(msg) {
  const el = document.getElementById('err');
  el.style.display = 'block';
  el.textContent = msg;
  throw new Error(msg);
}

function mat4Mul(out, a, b) {
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
    out[c * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
    out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
    out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
}

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

const dirFromAngles = (azDeg, elDeg) => {
  const a = azDeg * D2R, e = elDeg * D2R;
  return [Math.sin(a) * Math.cos(e), Math.sin(e), -Math.cos(a) * Math.cos(e)];
};

async function main() {
  if (!navigator.gpu) fail('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) fail('No WebGPU adapter found.');
  const hasTS = adapter.features.has('timestamp-query');
  const hasF32Filter = adapter.features.has('float32-filterable');
  const features = [];
  if (hasTS) features.push('timestamp-query');
  if (hasF32Filter) features.push('float32-filterable');
  else console.warn('float32-filterable unavailable: heights fall back to rgba16float and show 1 m contour steps');
  const HFMT = hasF32Filter ? 'rgba32float' : 'rgba16float';
  const H1FMT = hasF32Filter ? 'r32float' : 'r16float';
  const device = await adapter.requestDevice({ requiredFeatures: features });
  device.lost.then((i) => console.error('device lost: ' + i.message));
  device.onuncapturederror = (e) => console.error('GPU error: ' + e.error.message);

  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('webgpu');
  const swapFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: swapFormat, alphaMode: 'opaque' });
  const hudFps = document.getElementById('fps');
  hudFps.textContent = 'generating terrain…';

  async function shader(code, label) {
    const m = device.createShaderModule({ code, label });
    const info = await m.getCompilationInfo();
    let bad = false;
    for (const msg of info.messages) {
      if (msg.type === 'error') bad = true;
      if (msg.type !== 'info') {
        const line = code.split('\n')[msg.lineNum - 1];
        console.error(`[${label}] ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}\n    ${line}`);
      }
    }
    if (bad) fail('Shader compile error in ' + label + ' (see console)');
    return m;
  }
  async function computePipe(code, label) {
    const module = await shader(code, label);
    return device.createComputePipelineAsync({ label, layout: 'auto', compute: { module, entryPoint: 'main' } });
  }
  const mkBuf = (size, usage, label) => device.createBuffer({ size, usage, label });
  const bg = (pipe, entries, label) => device.createBindGroup({
    label, layout: pipe.getBindGroupLayout(0),
    entries: entries.map((r, i) => ({ binding: i, resource: r })),
  });
  const lvl = (t, m) => t.createView({ dimension: '2d-array', baseMipLevel: m, mipLevelCount: 1 });

  const [genPipe, derivPipe, hMipPipe, h1MipPipe, mMipPipe, probePipe, shadowPipe] = await Promise.all([
    computePipe(GEN, 'terrain gen'),
    computePipe(DERIV(HFMT, H1FMT), 'terrain deriv'),
    computePipe(HEIGHT_MIPS(HFMT), 'height mips'),
    computePipe(SCALAR_MIPS(H1FMT), 'scalar height mips'),
    computePipe(MAT_MIPS, 'material mips'),
    computePipe(PROBE, 'height probe'),
    computePipe(SHADOW_MAP(HFMT), 'shadow clipmap'),
  ]);

  const hTex = device.createTexture({
    label: 'height layers', size: [TEX, TEX, LAYERS], format: HFMT, mipLevelCount: TEX_MIPS,
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  });
  const hmTex = device.createTexture({
    label: 'march heights', size: [TEX, TEX, LAYERS], format: H1FMT, mipLevelCount: TEX_MIPS,
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_SRC,
  });
  const mTex = device.createTexture({
    label: 'material layers', size: [TEX, TEX, LAYERS], format: 'rgba8unorm', mipLevelCount: TEX_MIPS,
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  });
  const hRaw = device.createTexture({
    label: 'raw height', size: [TEX, TEX, LAYERS], format: 'r32float',
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  });
  const layerUBOs = LAYER_P.map((_, k) => {
    const b = mkBuf(16, GU.UNIFORM | GU.COPY_DST, 'layer ' + k);
    device.queue.writeBuffer(b, 0, new Uint32Array([k, 0, 0, 0]));
    return b;
  });
  for (let k = 0; k < LAYERS; k++) {
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(genPipe);
    cp.setBindGroup(0, bg(genPipe, [{ buffer: layerUBOs[k] }, lvl(hRaw, 0), lvl(mTex, 0)]));
    cp.dispatchWorkgroups(TEX / 8, TEX / 8, 1);
    cp.end();
    device.queue.submit([enc.finish()]);
  }
  {
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(derivPipe);
    const rawView = hRaw.createView({ dimension: '2d-array' });
    for (let k = 0; k < LAYERS; k++) {
      cp.setBindGroup(0, bg(derivPipe, [{ buffer: layerUBOs[k] }, rawView, lvl(hTex, 0), lvl(hmTex, 0)]));
      cp.dispatchWorkgroups(TEX / 8, TEX / 8, 1);
    }
    for (let m = 1; m < TEX_MIPS; m++) {
      const s = Math.max(1, TEX >> m);
      cp.setPipeline(hMipPipe);
      cp.setBindGroup(0, bg(hMipPipe, [lvl(hTex, m - 1), lvl(hTex, m)]));
      cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), LAYERS);
      cp.setPipeline(h1MipPipe);
      cp.setBindGroup(0, bg(h1MipPipe, [lvl(hmTex, m - 1), lvl(hmTex, m)]));
      cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), LAYERS);
      cp.setPipeline(mMipPipe);
      cp.setBindGroup(0, bg(mMipPipe, [lvl(mTex, m - 1), lvl(mTex, m)]));
      cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), LAYERS);
    }
    cp.end();
    device.queue.submit([enc.finish()]);
  }
  const layerMeans = new Float32Array(LAYERS);
  {
    const meanBuf = mkBuf(256 * LAYERS, GU.MAP_READ | GU.COPY_DST, 'layer means');
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: hmTex, mipLevel: TEX_MIPS - 1 },
      { buffer: meanBuf, bytesPerRow: 256, rowsPerImage: 1 }, [1, 1, LAYERS]);
    device.queue.submit([enc.finish()]);
    await meanBuf.mapAsync(GPUMapMode.READ);
    const raw = meanBuf.getMappedRange();
    const is16 = H1FMT === 'r16float';
    for (let k = 0; k < LAYERS; k++) {
      layerMeans[k] = is16 ? halfToFloat(new Uint16Array(raw, k * 256, 1)[0]) : new Float32Array(raw, k * 256, 1)[0];
    }
    meanBuf.unmap();
    meanBuf.destroy();
  }
  hRaw.destroy();

  const MSAA = 4;
  const HDR = 'rgba16float';
  const [terrainMod, skyMod, downMod, upMod, compMod] = await Promise.all([
    shader(TERRAIN, 'terrain'), shader(SKY, 'sky'), shader(BLOOM_DOWN, 'bloom down'),
    shader(BLOOM_UP, 'bloom up'), shader(COMPOSITE, 'composite'),
  ]);
  const terrainPipe = await device.createRenderPipelineAsync({
    label: 'terrain', layout: 'auto',
    vertex: {
      module: terrainMod, entryPoint: 'vs',
      buffers: [{ arrayStride: 16, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }],
    },
    fragment: { module: terrainMod, entryPoint: 'fs', targets: [{ format: HDR }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    multisample: { count: MSAA },
  });
  const skyPipe = await device.createRenderPipelineAsync({
    label: 'sky', layout: 'auto',
    vertex: { module: skyMod, entryPoint: 'vs' },
    fragment: { module: skyMod, entryPoint: 'fs', targets: [{ format: HDR }] },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'equal' },
    multisample: { count: MSAA },
  });
  const downPipe = await device.createRenderPipelineAsync({
    label: 'bloom down', layout: 'auto',
    vertex: { module: downMod, entryPoint: 'vs' },
    fragment: { module: downMod, entryPoint: 'fs', targets: [{ format: HDR }] },
  });
  const additive = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
  const upPipe = await device.createRenderPipelineAsync({
    label: 'bloom up', layout: 'auto',
    vertex: { module: upMod, entryPoint: 'vs' },
    fragment: { module: upMod, entryPoint: 'fs', targets: [{ format: HDR, blend: { color: additive, alpha: additive } }] },
  });
  const compPipe = await device.createRenderPipelineAsync({
    label: 'composite', layout: 'auto',
    vertex: { module: compMod, entryPoint: 'vs' },
    fragment: { module: compMod, entryPoint: 'fs', targets: [{ format: swapFormat }] },
  });

  const sampRepeat = device.createSampler({
    magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    addressModeU: 'repeat', addressModeV: 'repeat', maxAnisotropy: 16,
  });
  const sampClamp = device.createSampler({
    magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
  });
  const frameUBO = mkBuf(544, GU.UNIFORM | GU.COPY_DST, 'frame ubo');
  const hView = hTex.createView({ dimension: '2d-array' });
  const hmView = hmTex.createView({ dimension: '2d-array' });
  const shTex = device.createTexture({
    label: 'shadow clipmaps', size: [SH_RES, SH_RES, SH_LEVELS * 2], format: HFMT,
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_SRC,
  });
  const bgTerrain = bg(terrainPipe, [{ buffer: frameUBO }, hView, mTex.createView({ dimension: '2d-array' }), sampRepeat,
    shTex.createView({ dimension: '2d-array' }), sampClamp, hmView], 'terrain');
  const shJobUBO = mkBuf(80, GU.UNIFORM | GU.COPY_DST, 'shadow job');
  const shJobF = new Float32Array(20);
  const shBGs = [];
  for (let i = 0; i < SH_LEVELS * 2; i++) {
    const v = shTex.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 });
    shBGs.push(bg(shadowPipe, [{ buffer: frameUBO }, { buffer: shJobUBO }, hmView, sampRepeat, v, hView], 'shadow ' + i));
  }
  const shLv = SH_SIZE.map((size) => ({ size, texel: size / SH_RES, front: -1, cx: 0, cz: 0, az: 0 }));
  let shJob = null;

  // One clipmap level at a time is rebuilt, SH_ROWS rows per frame, into the level's back layer;
  // it becomes visible when complete. A level is due when the camera has moved an eighth of its
  // width or the sun azimuth changed.
  function encodeShadowJob(cp) {
    if (!shJob) {
      let best = -1, bestScore = 1;
      for (let l = 0; l < SH_LEVELS; l++) {
        const L = shLv[l];
        let score;
        if (L.front < 0) score = 100 - l;
        else {
          const moved = Math.max(Math.abs(cam.x - L.cx), Math.abs(cam.z - L.cz)) / L.size;
          const daz = Math.abs(P.sunAzim - L.az);
          score = Math.max(moved * 8, daz > 0.2 ? 1 + daz : 0);
        }
        if (score >= bestScore) { bestScore = score; best = l; }
      }
      if (best < 0) return;
      const L = shLv[best];
      shJob = {
        l: best, row: 0, az: P.sunAzim,
        cx: Math.round(cam.x / L.texel) * L.texel, cz: Math.round(cam.z / L.texel) * L.texel,
        back: L.front === best * 2 ? best * 2 + 1 : best * 2,
      };
    }
    const L = shLv[shJob.l];
    for (let k = 0; k < LAYERS; k++) {
      const per = LAYER_P[k], c = Math.cos(LAYER_ROT[k]), s = Math.sin(LAYER_ROT[k]);
      const lu = (c * shJob.cx - s * shJob.cz + LAYER_OFF[k][0]) / per;
      const lv = (s * shJob.cx + c * shJob.cz + LAYER_OFF[k][1]) / per;
      shJobF[2 * k] = lu - Math.floor(lu); shJobF[2 * k + 1] = lv - Math.floor(lv);
    }
    shJobF[12] = -L.size / 2; shJobF[13] = -L.size / 2; shJobF[14] = L.texel; shJobF[15] = shJob.row;
    shJobF[16] = Math.sin(shJob.az * D2R); shJobF[17] = -Math.cos(shJob.az * D2R);
    device.queue.writeBuffer(shJobUBO, 0, shJobF);
    cp.setPipeline(shadowPipe);
    cp.setBindGroup(0, shBGs[shJob.back]);
    cp.dispatchWorkgroups(SH_RES / 8, SH_ROWS / 8);
    shJob.row += SH_ROWS;
    if (shJob.row >= SH_RES) {
      L.front = shJob.back; L.cx = shJob.cx; L.cz = shJob.cz; L.az = shJob.az;
      shJob = null;
    }
  }
  const bgSky = bg(skyPipe, [{ buffer: frameUBO }], 'sky');

  const probeOut = mkBuf(16, GU.STORAGE | GU.COPY_SRC, 'probe out');
  const probeRead = mkBuf(16, GU.MAP_READ | GU.COPY_DST, 'probe read');
  const probeUBO = mkBuf(16, GU.UNIFORM | GU.COPY_DST, 'probe ubo');
  const probeF = new Float32Array(4);
  const bgProbe = bg(probePipe, [{ buffer: frameUBO }, hmView, sampRepeat, { buffer: probeOut }, { buffer: probeUBO }], 'probe');
  let probePending = false, groundH = 0, groundTop = 0, haveGround = false;
  let velX = 0, velZ = 0;
  const onProbeFail = () => { probePending = false; };

  const GRID = 32;
  const LEAF = 2.0;
  const R0 = 8 * LEAF;
  const LODS = 17;
  const MAX_DIST = 150000;
  const MAX_NODES = 8192;
  const HMIN = -5000, HMAX = 4500;
  const idx = new Uint16Array(GRID * GRID * 6);
  {
    let k = 0;
    for (let j = 0; j < GRID; j++) {
      for (let i = 0; i < GRID; i++) {
        const a = i + j * (GRID + 1), b = a + 1, c = a + GRID + 1, d = c + 1;
        idx[k++] = a; idx[k++] = c; idx[k++] = d;
        idx[k++] = a; idx[k++] = d; idx[k++] = b;
      }
    }
  }
  const idxBuf = mkBuf(idx.byteLength, GU.INDEX | GU.COPY_DST, 'grid idx');
  device.queue.writeBuffer(idxBuf, 0, idx);
  const instData = new Float32Array(MAX_NODES * 4);
  const instBuf = mkBuf(instData.byteLength, GU.VERTEX | GU.COPY_DST, 'instances');

  const BLOOM_LEVELS = 8;
  const linSamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  const compUBO = mkBuf(32, GU.UNIFORM | GU.COPY_DST, 'composite ubo');
  const compData = new Float32Array(8);
  const bloomUBOs = [];
  for (let i = 0; i < BLOOM_LEVELS * 2; i++) bloomUBOs.push(mkBuf(16, GU.UNIFORM | GU.COPY_DST, 'bloom ubo'));

  let W = 0, H = 0;
  let colorMS, depthMS, hdrTex, bloomTex;
  const mainPass = {
    colorAttachments: [{ view: null, resolveTarget: null, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'discard' }],
    depthStencilAttachment: { view: null, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' },
  };
  const compPass = {
    colorAttachments: [{ view: null, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
    timestampWrites: undefined,
  };
  let downPasses = [], upPasses = [], downBGs = [], upBGs = [], bgComp = null;

  function resize() {
    const fw = Number(qs.get('w')) || 0, fh = Number(qs.get('h')) || 0;
    const dpr = (window.devicePixelRatio || 1) * P.scale;
    const w = fw || Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = fh || Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (w === W && h === H) return;
    W = w; H = h;
    canvas.width = W; canvas.height = H;
    for (const t of [colorMS, depthMS, hdrTex, bloomTex]) t && t.destroy();
    colorMS = device.createTexture({ size: [W, H], format: HDR, sampleCount: MSAA, usage: TU.RENDER_ATTACHMENT });
    depthMS = device.createTexture({ size: [W, H], format: 'depth32float', sampleCount: MSAA, usage: TU.RENDER_ATTACHMENT });
    hdrTex = device.createTexture({ size: [W, H], format: HDR, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    const bw = Math.max(1, W >> 1), bh = Math.max(1, H >> 1);
    bloomTex = device.createTexture({ size: [bw, bh], format: HDR, mipLevelCount: BLOOM_LEVELS, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    mainPass.colorAttachments[0].view = colorMS.createView();
    mainPass.colorAttachments[0].resolveTarget = hdrTex.createView();
    mainPass.depthStencilAttachment.view = depthMS.createView();
    const hdrView = hdrTex.createView();
    const bv = [];
    for (let m = 0; m < BLOOM_LEVELS; m++) bv.push(bloomTex.createView({ baseMipLevel: m, mipLevelCount: 1 }));
    downPasses = []; upPasses = []; downBGs = []; upBGs = [];
    const tmp = new Float32Array(4);
    for (let m = 0; m < BLOOM_LEVELS; m++) {
      const srcW = m === 0 ? W : Math.max(1, bw >> (m - 1));
      const srcH = m === 0 ? H : Math.max(1, bh >> (m - 1));
      tmp[0] = 1 / srcW; tmp[1] = 1 / srcH; tmp[2] = m === 0 ? 1 : 0; tmp[3] = m === 0 ? BLOOM_CLAMP : 0;
      device.queue.writeBuffer(bloomUBOs[m], 0, tmp);
      downBGs.push(bg(downPipe, [m === 0 ? hdrView : bv[m - 1], linSamp, { buffer: bloomUBOs[m] }]));
      downPasses.push({ colorAttachments: [{ view: bv[m], clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    }
    for (let m = BLOOM_LEVELS - 1; m > 0; m--) {
      const sw = Math.max(1, bw >> m), sh = Math.max(1, bh >> m);
      tmp[0] = 1 / sw; tmp[1] = 1 / sh; tmp[2] = 1.0; tmp[3] = 1.0;
      const ubo = bloomUBOs[BLOOM_LEVELS + m];
      device.queue.writeBuffer(ubo, 0, tmp);
      upBGs.push(bg(upPipe, [bv[m], linSamp, { buffer: ubo }]));
      upPasses.push({ colorAttachments: [{ view: bv[m - 1], loadOp: 'load', storeOp: 'store' }] });
    }
    bgComp = bg(compPipe, [hdrView, bv[0], linSamp, { buffer: compUBO }]);
  }

  let tsQuery = null, tsResolve = null, tsRead = null, tsPending = false, gpuMs = 0;
  let computeDesc = {};
  if (hasTS) {
    tsQuery = device.createQuerySet({ type: 'timestamp', count: 2 });
    tsResolve = mkBuf(16, GU.QUERY_RESOLVE | GU.COPY_SRC, 'ts resolve');
    tsRead = mkBuf(16, GU.MAP_READ | GU.COPY_DST, 'ts read');
    computeDesc = { timestampWrites: { querySet: tsQuery, beginningOfPassWriteIndex: 0 } };
    compPass.timestampWrites = { querySet: tsQuery, endOfPassWriteIndex: 1 };
  }
  const onTsMapped = () => {
    const t = new BigUint64Array(tsRead.getMappedRange());
    const ms = Number(t[1] - t[0]) / 1e6;
    if (ms > 0 && ms < 1000) gpuMs = gpuMs ? gpuMs * 0.8 + ms * 0.2 : ms;
    tsRead.unmap();
    tsPending = false;
  };
  const onTsFail = () => { tsPending = false; };

  const cam = { x: -14000, y: 0, z: -9000, yaw: -90 * D2R, pitch: 5 * D2R, fovY: P.fov * D2R, speed: 8, vy: 0 };
  let startAlt = EYE_HEIGHT + 0.1;
  if (qs.get('cam')) {
    const v = qs.get('cam').split(',').map(Number);
    cam.x = v[0] || 0; startAlt = v[1] || startAlt; cam.z = v[2] || 0;
    cam.yaw = (v[3] || 0) * D2R; cam.pitch = (v[4] || 0) * D2R;
  }
  if (!qs.has('scale') && innerWidth * innerHeight * devicePixelRatio ** 2 > 4.5e6) P.scale = 0.7;
  for (const k of Object.keys(P)) if (qs.has(k)) P[k] = Number(qs.get(k));
  cam.fovY = P.fov * D2R;
  let walking = qs.get('walk') === '1';
  let zoomed = false;
  let grounded = false;

  const keys = Object.create(null);
  addEventListener('keydown', (e) => {
    keys[e.code] = true;
    if (e.code === 'KeyH') panel.classList.toggle('hidden');
    if (e.code === 'KeyG') { walking = !walking; cam.vy = 0; }
    if (e.code === 'KeyZ') zoomed = !zoomed;
  });
  addEventListener('keyup', (e) => { keys[e.code] = false; });
  addEventListener('blur', () => { for (const k in keys) keys[k] = false; });
  canvas.addEventListener('click', () => { if (document.pointerLockElement !== canvas) canvas.requestPointerLock(); });
  let dragging = false;
  canvas.addEventListener('mousedown', (e) => { if (e.button === 2) dragging = true; });
  addEventListener('mouseup', () => { dragging = false; });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  addEventListener('mousemove', (e) => {
    if (document.pointerLockElement === canvas || dragging) {
      const k = 0.0022 * cam.fovY / (58 * D2R);
      cam.yaw += e.movementX * k;
      cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch - e.movementY * k));
    }
  });
  addEventListener('wheel', (e) => {
    cam.speed = Math.max(0.5, Math.min(20000, cam.speed * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
  }, { passive: true });

  function updateCamera(dt) {
    const fovTarget = (zoomed ? 6 : P.fov) * D2R;
    cam.fovY += (fovTarget - cam.fovY) * Math.min(1, dt * 8);
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const fwd = (keys.KeyW ? 1 : 0) - (keys.KeyS ? 1 : 0);
    const str = (keys.KeyD ? 1 : 0) - (keys.KeyA ? 1 : 0);
    const boost = keys.ShiftLeft || keys.ShiftRight ? 5 : 1;
    if (walking) {
      const v = 2.2 * boost * dt;
      cam.x += (fwd * sy + str * cy) * v;
      cam.z += (-fwd * cy + str * sy) * v;
      const floor = Math.max(groundH + EYE_HEIGHT, groundTop + CLEARANCE);
      if (grounded && (keys.Space || keys.KeyE)) { cam.vy = 2.4; grounded = false; }
      cam.vy -= LUNAR_G * dt;
      cam.y += cam.vy * dt;
      if (cam.y <= floor) { cam.y = floor; cam.vy = 0; grounded = true; }
      return;
    }
    const up = (keys.KeyE || keys.Space ? 1 : 0) - (keys.KeyQ || keys.KeyC ? 1 : 0);
    const v = cam.speed * boost * dt;
    cam.x += (fwd * cp * sy + str * cy) * v;
    cam.y += (fwd * sp + up) * v;
    cam.z += (-fwd * cp * cy + str * sy) * v;
    cam.y = Math.min(60000, cam.y);
    if (haveGround) cam.y = Math.max(cam.y, groundTop + CLEARANCE);
  }

  const ranges = new Float64Array(LODS);
  for (let l = 0; l < LODS; l++) ranges[l] = R0 * Math.pow(2, l);
  const planes = new Float64Array(16);
  let nodeCount = 0, ccx = 0, ccy = 0, ccz = 0, lodDy = 0;

  function sphereHit(x0, z0, size, r) {
    const dx = Math.max(x0 - ccx, 0, ccx - (x0 + size));
    const dz = Math.max(z0 - ccz, 0, ccz - (z0 + size));
    return dx * dx + lodDy * lodDy + dz * dz <= r * r;
  }
  function inFrustum(x0, z0, size) {
    const minx = x0 - ccx, maxx = minx + size;
    const minz = z0 - ccz, maxz = minz + size;
    const fx = Math.max(-minx, maxx), fz = Math.max(-minz, maxz);
    const miny = HMIN - ccy - (fx * fx + fz * fz) / (2 * MOON_R), maxy = HMAX - ccy;
    for (let p = 0; p < 16; p += 4) {
      const a = planes[p], b = planes[p + 1], c = planes[p + 2], d = planes[p + 3];
      const px = a > 0 ? maxx : minx, py = b > 0 ? maxy : miny, pz = c > 0 ? maxz : minz;
      if (a * px + b * py + c * pz + d < 0) return false;
    }
    return true;
  }
  function addNode(x0, z0, size, lod) {
    if (nodeCount >= MAX_NODES) return;
    const o = nodeCount * 4;
    instData[o] = x0 - ccx; instData[o + 1] = z0 - ccz; instData[o + 2] = size; instData[o + 3] = lod;
    nodeCount++;
  }
  function selectNode(x0, z0, size, lod) {
    if (!sphereHit(x0, z0, size, ranges[lod])) return false;
    if (!sphereHit(x0, z0, size, MAX_DIST) || !inFrustum(x0, z0, size)) return true;
    if (lod === 0 || !sphereHit(x0, z0, size, ranges[lod - 1])) { addNode(x0, z0, size, lod); return true; }
    const h = size * 0.5;
    for (let i = 0; i < 4; i++) {
      const nx = x0 + (i & 1) * h, nz = z0 + (i >> 1) * h;
      if (!selectNode(nx, nz, h, lod - 1)) {
        if (sphereHit(nx, nz, h, MAX_DIST) && inFrustum(nx, nz, h)) addNode(nx, nz, h, lod - 1);
      }
    }
    return true;
  }
  function selectAll() {
    nodeCount = 0;
    ccx = cam.x; ccy = cam.y; ccz = cam.z;
    lodDy = Math.max(groundH - 300 - ccy, 0, ccy - (groundH + 300));
    const m = viewProj;
    for (let i = 0; i < 4; i++) {
      const r = i >> 1, s = (i & 1) ? -1 : 1;
      planes[i * 4] = m[3] + s * m[r];
      planes[i * 4 + 1] = m[7] + s * m[4 + r];
      planes[i * 4 + 2] = m[11] + s * m[8 + r];
      planes[i * 4 + 3] = m[15] + s * m[12 + r];
    }
    const top = LODS - 1;
    const ts = LEAF * Math.pow(2, top);
    const x0 = Math.floor((ccx - MAX_DIST) / ts), x1 = Math.floor((ccx + MAX_DIST) / ts);
    const z0 = Math.floor((ccz - MAX_DIST) / ts), z1 = Math.floor((ccz + MAX_DIST) / ts);
    for (let zi = z0; zi <= z1; zi++) {
      for (let xi = x0; xi <= x1; xi++) selectNode(xi * ts, zi * ts, ts, top);
    }
  }

  const frameF = new Float32Array(136);
  frameF.set(layerMeans, 116);
  const proj = new Float32Array(16), view = new Float32Array(16), viewProj = new Float32Array(16);
  const earthAxis = (() => {
    const v = [0.2, 0.92, 0.33];
    const l = Math.hypot(...v);
    return v.map((x) => x / l);
  })();
  let time = 0, skyTime = 0, frameIndex = 0;

  function writeFrame() {
    const aspect = W / H;
    const tanY = Math.tan(cam.fovY * 0.5), tanX = tanY * aspect;
    const near = 0.03;
    proj.fill(0);
    proj[0] = 1 / tanX; proj[5] = 1 / tanY; proj[11] = -1; proj[14] = near;
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const fx = cp * sy, fy = sp, fz = -cp * cy;
    const rx = cy, ry = 0, rz = sy;
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    view.fill(0);
    view[0] = rx; view[4] = ry; view[8] = rz;
    view[1] = ux; view[5] = uy; view[9] = uz;
    view[2] = -fx; view[6] = -fy; view[10] = -fz;
    view[15] = 1;
    mat4Mul(viewProj, proj, view);
    const f = frameF;
    f.set(viewProj, 0);
    f[16] = 0; f[17] = cam.y; f[18] = 0; f[19] = time;
    f[20] = rx * tanX; f[21] = ry * tanX; f[22] = rz * tanX;
    f[24] = ux * tanY; f[25] = uy * tanY; f[26] = uz * tanY;
    f[28] = fx; f[29] = fy; f[30] = fz;
    const sd = dirFromAngles(P.sunAzim, P.sunElev);
    f[32] = sd[0]; f[33] = sd[1]; f[34] = sd[2]; f[35] = 1.0;
    const ed = dirFromAngles(P.earthAzim, P.earthElev);
    f[36] = ed[0]; f[37] = ed[1]; f[38] = ed[2]; f[39] = EARTH_ANG_R;
    f[40] = W; f[41] = H; f[42] = 1 / W; f[43] = 1 / H;
    for (let k = 0; k < LAYERS; k++) {
      const per = LAYER_P[k], c = Math.cos(LAYER_ROT[k]), s = Math.sin(LAYER_ROT[k]);
      f[44 + k] = per;
      f[52 + 2 * k] = c; f[53 + 2 * k] = s;
      const lu = (c * cam.x - s * cam.z + LAYER_OFF[k][0]) / per;
      const lv = (s * cam.x + c * cam.z + LAYER_OFF[k][1]) / per;
      f[64 + 2 * k] = lu - Math.floor(lu); f[65 + 2 * k] = lv - Math.floor(lv);
    }
    f[76] = R0; f[77] = GRID; f[78] = groundH - 300; f[79] = groundH + 300;
    f[80] = 0; f[81] = 0; f[82] = frameIndex; f[83] = DBG;
    f[84] = P.albedo; f[85] = P.bounce; f[86] = P.earthshine; f[87] = Math.round(P.shadowSteps);
    f[88] = P.stars; f[89] = P.milkyWay; f[90] = skyTime * (2 * Math.PI / SIDEREAL_DAY_S); f[91] = skyTime / 3600;
    const skyAngle = skyTime * (2 * Math.PI / SYNODIC_MONTH_S);
    for (let c = 0; c < 3; c++) {
      const col = rotateAbout([c === 0 ? 1 : 0, c === 1 ? 1 : 0, c === 2 ? 1 : 0], POLE, skyAngle);
      f[124 + 4 * c] = col[0]; f[125 + 4 * c] = col[1]; f[126 + 4 * c] = col[2];
    }
    f[92] = earthAxis[0]; f[93] = earthAxis[1]; f[94] = earthAxis[2]; f[95] = P.earthBright;
    for (let l = 0; l < SH_LEVELS; l++) {
      const L = shLv[l];
      f[96 + 4 * l] = L.cx - cam.x; f[97 + 4 * l] = L.cz - cam.z; f[98 + 4 * l] = L.size; f[99 + 4 * l] = L.front;
    }
    device.queue.writeBuffer(frameUBO, 0, f);
  }

  const panel = document.getElementById('panel');
  const SLIDERS = [
    ['sunElev', 'Sun elevation (deg)', -5, 60, 0.1],
    ['sunAzim', 'Sun azimuth (deg)', -180, 180, 0.5],
    ['earthElev', 'Earth elevation (deg)', -10, 80, 0.1],
    ['earthAzim', 'Earth azimuth (deg)', -180, 180, 0.5],
    ['earthBright', 'Earth brightness', 0, 2, 0.01],
    ['exposure', 'Exposure (EV)', -2, 18, 0.05],
    ['timeLapse', 'Sky time lapse (x)', 0, 20000, 10],
    ['sunMoves', 'Sun follows sky (0/1)', 0, 1, 1],
    ['albedo', 'Regolith albedo', 0.04, 0.3, 0.001],
    ['bounce', 'Bounce light', 0, 3, 0.01],
    ['earthshine', 'Earthshine', 0, 50, 0.1],
    ['stars', 'Stars', 0, 5, 0.01],
    ['milkyWay', 'Milky Way', 0, 5, 0.01],
    ['shadowSteps', 'Near shadow steps', 4, 40, 1],
    ['bloom', 'Bloom', 0, 0.3, 0.001],
    ['fov', 'Field of view (deg)', 10, 100, 0.5],
    ['scale', 'Resolution scale', 0.25, 1, 0.05],
  ];
  const sliderEls = {};
  for (const [key, label, min, max, step] of SLIDERS) {
    const row = document.createElement('label');
    const name = document.createElement('span'); name.textContent = label;
    const val = document.createElement('output'); val.textContent = String(P[key]);
    const inp = document.createElement('input');
    Object.assign(inp, { type: 'range', min, max, step, value: P[key] });
    inp.addEventListener('input', () => { P[key] = Number(inp.value); val.textContent = inp.value; });
    sliderEls[key] = { inp, val };
    row.append(name, inp, val);
    panel.appendChild(row);
  }
  if (qs.get('ui') === '0') { document.getElementById('hud').style.display = 'none'; panel.classList.add('hidden'); }

  function advanceSky(dt) {
    const dSky = dt * P.timeLapse;
    skyTime += dSky;
    if (!P.sunMoves || dSky === 0) return;
    const v = rotateAbout(dirFromAngles(P.sunAzim, P.sunElev), POLE, -dSky * (2 * Math.PI / SYNODIC_MONTH_S));
    P.sunElev = Math.asin(Math.max(-1, Math.min(1, v[1]))) / D2R;
    P.sunAzim = Math.atan2(v[0], -v[2]) / D2R;
    for (const key of ['sunElev', 'sunAzim']) {
      const el = sliderEls[key];
      if (!el) continue;
      el.inp.value = P[key];
      el.val.textContent = P[key].toFixed(1);
    }
  }

  const submitList = [null];
  let last = performance.now();
  let fpsAcc = 0, fpsN = 0, fpsT = 0;

  function onProbe() {
    const h = new Float32Array(probeRead.getMappedRange(0, 16));
    groundH = h[0];
    groundTop = h[1];
    probeRead.unmap();
    probePending = false;
    if (!haveGround) {
      haveGround = true;
      cam.y = groundH + startAlt;
    }
  }

  function frame(now) {
    requestAnimationFrame(frame);
    resize();
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    fpsAcc += dt; fpsN++; fpsT += dt;
    if (fpsT > 0.5) {
      const fps = fpsN / fpsAcc;
      hudFps.textContent = `${fps.toFixed(0)} fps  ${(1000 / fps).toFixed(2)} ms` +
        (hasTS ? `  |  GPU ${gpuMs.toFixed(2)} ms` : '') + `  |  ${nodeCount} nodes  ${W}x${H}` +
        `  |  ${(cam.y - groundH).toFixed(1)} m above ground  ${walking ? 'walking' : 'flying ' + cam.speed.toFixed(1) + ' m/s'}`;
      fpsAcc = 0; fpsN = 0; fpsT = 0;
    }
    const px = cam.x, pz = cam.z;
    if (haveGround) updateCamera(dt);
    if (dt > 0) {
      const k = Math.min(1, dt * 10);
      velX += ((cam.x - px) / dt - velX) * k;
      velZ += ((cam.z - pz) / dt - velZ) * k;
    }
    time += dt;
    advanceSky(dt);
    frameIndex++;

    writeFrame();
    selectAll();
    if (nodeCount) device.queue.writeBuffer(instBuf, 0, instData, 0, nodeCount * 4);

    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass(computeDesc);
    const speed = Math.hypot(velX, velZ);
    probeF[0] = velX * 0.25; probeF[1] = velZ * 0.25; probeF[2] = 0.6 + speed * 0.12;
    device.queue.writeBuffer(probeUBO, 0, probeF);
    cp.setPipeline(probePipe);
    cp.setBindGroup(0, bgProbe);
    cp.dispatchWorkgroups(1);
    if (haveGround) encodeShadowJob(cp);
    cp.end();

    const rp = enc.beginRenderPass(mainPass);
    if (nodeCount) {
      rp.setPipeline(terrainPipe);
      rp.setBindGroup(0, bgTerrain);
      rp.setVertexBuffer(0, instBuf);
      rp.setIndexBuffer(idxBuf, 'uint16');
      rp.drawIndexed(GRID * GRID * 6, nodeCount);
    }
    rp.setPipeline(skyPipe);
    rp.setBindGroup(0, bgSky);
    rp.draw(3);
    rp.end();

    for (let m = 0; m < BLOOM_LEVELS; m++) {
      const p = enc.beginRenderPass(downPasses[m]);
      p.setPipeline(downPipe);
      p.setBindGroup(0, downBGs[m]);
      p.draw(3);
      p.end();
    }
    for (let i = 0; i < upPasses.length; i++) {
      const p = enc.beginRenderPass(upPasses[i]);
      p.setPipeline(upPipe);
      p.setBindGroup(0, upBGs[i]);
      p.draw(3);
      p.end();
    }

    compData[0] = Math.pow(2, P.exposure); compData[1] = P.bloom; compData[2] = frameIndex % 64; compData[3] = 0.10;
    compData[4] = 1.0; compData[5] = 1.04; compData[6] = 0; compData[7] = 0;
    device.queue.writeBuffer(compUBO, 0, compData);
    compPass.colorAttachments[0].view = ctx.getCurrentTexture().createView();
    const pc = enc.beginRenderPass(compPass);
    pc.setPipeline(compPipe);
    pc.setBindGroup(0, bgComp);
    pc.draw(3);
    pc.end();

    let doProbe = false;
    if (!probePending) {
      enc.copyBufferToBuffer(probeOut, 0, probeRead, 0, 16);
      doProbe = true;
    }
    let doTs = false;
    if (hasTS && !tsPending && (frameIndex & 7) === 0) {
      enc.resolveQuerySet(tsQuery, 0, 2, tsResolve, 0);
      enc.copyBufferToBuffer(tsResolve, 0, tsRead, 0, 16);
      doTs = true;
    }
    submitList[0] = enc.finish();
    device.queue.submit(submitList);
    if (doProbe) { probePending = true; probeRead.mapAsync(GPUMapMode.READ).then(onProbe, onProbeFail); }
    if (doTs) { tsPending = true; tsRead.mapAsync(GPUMapMode.READ).then(onTsMapped, onTsFail); }
  }

  if (DBG) window.__moon = { device, shTex, shLv, cam, P, get groundH() { return groundH; } };
  resize();
  addEventListener('resize', resize);
  requestAnimationFrame(frame);
}

main().catch((e) => { console.error('init failed', e); fail(String(e && e.stack || e)); });
