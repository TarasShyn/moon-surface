export const TEX = 1024;
export const LAYERS = 6;
export const TEX_MIPS = 11;
export const LAYER_P = [131072.0, 15731.0, 1891.0, 227.3, 27.31, 3.282];
export const LAYER_ROT = [0.0, 0.61, 1.93, 2.71, 4.07, 5.33];
export const LAYER_OFF = [
  [65536.0, 74536.0], [5023.0, 9311.0], [711.0, 1203.0], [61.0, 147.0], [7.3, 19.1], [1.13, 2.71],
];
export const MOON_R = 1737400.0;
export const HERO_R = 24000.0;
export const SH_RES = 512;
export const SH_LEVELS = 5;

export const wgslF = (x) => {
  const s = String(x);
  return s.includes('.') || s.includes('e') ? s : s + '.0';
};

// Float offsets in main.js writeFrame() must match this layout.
export const FRAME_WGSL = /* wgsl */ `
struct Frame {
  viewProj:  mat4x4f,
  camPos:    vec4f,
  camRight:  vec4f,
  camUp:     vec4f,
  camFwd:    vec4f,
  sunDir:    vec4f,
  earthDir:  vec4f,
  screen:    vec4f,
  layerP:    array<vec4f, 2>,
  layerRot:  array<vec4f, 3>,
  layerOff:  array<vec4f, 3>,
  lodP:      vec4f,
  post:      vec4f,
  shade:     vec4f,
  sky:       vec4f,
  earthAxis: vec4f,
  shadowLvl: array<vec4f, 5>,
  layerMean: array<vec4f, 2>,
  skyRot: mat3x3f,
};
const PI: f32 = 3.14159265358979;
const MOON_R: f32 = ${wgslF(MOON_R)};
const SUN_ANG_R: f32 = 0.00465;
`;

export const LAYER_FUNCS = /* wgsl */ `
fn layerPeriod(k: u32) -> f32 { return F.layerP[k >> 2u][k & 3u]; }
fn layerCS(k: u32) -> vec2f {
  let v = F.layerRot[k >> 1u];
  return select(v.xy, v.zw, (k & 1u) == 1u);
}
fn layerUV(rel: vec2f, k: u32) -> vec2f {
  let cs = layerCS(k);
  let v = F.layerOff[k >> 1u];
  let o = select(v.xy, v.zw, (k & 1u) == 1u);
  return o + vec2f(cs.x * rel.x - cs.y * rel.y, cs.y * rel.x + cs.x * rel.y) / layerPeriod(k);
}
fn gradToWorld(g: vec2f, k: u32) -> vec2f {
  let cs = layerCS(k);
  return vec2f(cs.x * g.x + cs.y * g.y, -cs.y * g.x + cs.x * g.y);
}
`;

export const NOISE_WGSL = /* wgsl */ `
fn pcgHash(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hash4(c: vec2i, s: u32) -> vec4f {
  let h0 = pcgHash(u32(c.x) * 1664525u + pcgHash(u32(c.y) * 22695477u + s));
  let h1 = pcgHash(h0 + 0x9e3779b9u);
  let h2 = pcgHash(h1 + 0x85ebca6bu);
  let h3 = pcgHash(h2 + 0xc2b2ae35u);
  return vec4f(f32(h0), f32(h1), f32(h2), f32(h3)) * (1.0 / 4294967296.0);
}
fn hashGrad(i: vec2i) -> vec2f {
  let h = pcgHash(u32(i.x) * 1664525u + pcgHash(u32(i.y) + 1013904223u));
  let a = f32(h) * (6.28318530718 / 4294967296.0);
  return vec2f(cos(a), sin(a));
}
fn gnoise(p: vec2f) -> f32 {
  let i = vec2i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let a = dot(hashGrad(i), f);
  let b = dot(hashGrad(i + vec2i(1, 0)), f - vec2f(1.0, 0.0));
  let c = dot(hashGrad(i + vec2i(0, 1)), f - vec2f(0.0, 1.0));
  let d = dot(hashGrad(i + vec2i(1, 1)), f - vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn pnoise(p: vec2f, period: i32) -> f32 {
  let i = vec2i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let i00 = (i % period + period) % period;
  let i10 = ((i + vec2i(1, 0)) % period + period) % period;
  let i01 = ((i + vec2i(0, 1)) % period + period) % period;
  let i11 = ((i + vec2i(1, 1)) % period + period) % period;
  let a = dot(hashGrad(i00), f);
  let b = dot(hashGrad(i10), f - vec2f(1.0, 0.0));
  let c = dot(hashGrad(i01), f - vec2f(0.0, 1.0));
  let d = dot(hashGrad(i11), f - vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn pfbm(p: vec2f, period: i32, oct: i32) -> f32 {
  var s = 0.0; var a = 0.5; var q = p; var per = period;
  for (var o = 0; o < oct; o++) {
    s += a * pnoise(q, per);
    q *= 2.0; per *= 2; a *= 0.5;
  }
  return s;
}
fn pfbmGain(p: vec2f, period: i32, oct: i32, gain: f32) -> f32 {
  var s = 0.0; var a = 0.5; var q = p; var per = period;
  for (var o = 0; o < oct; o++) {
    s += a * pnoise(q, per);
    q *= 2.0; per *= 2; a *= gain;
  }
  return s;
}
fn pridged(p: vec2f, period: i32, oct: i32) -> f32 {
  var s = 0.0; var a = 0.5; var q = p; var per = period; var w = 1.0;
  for (var o = 0; o < oct; o++) {
    var n = 1.0 - abs(pnoise(q, per) * 1.6);
    n = n * n;
    s += a * n * w;
    w = clamp(n * 1.6, 0.0, 1.0);
    q *= 2.0; per *= 2; a *= 0.5;
  }
  return s;
}
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}
fn smax(a: f32, b: f32, k: f32) -> f32 { return -smin(-a, -b, k); }

fn hash3f(i: vec3i) -> f32 {
  let h = pcgHash((u32(i.x) * 73856093u) ^ pcgHash((u32(i.y) * 19349663u) ^ pcgHash(u32(i.z) * 83492791u)));
  return f32(h) * (1.0 / 4294967296.0);
}
fn vnoise3(p: vec3f) -> f32 {
  let i = vec3i(floor(p));
  let f = p - floor(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash3f(i);
  let b = hash3f(i + vec3i(1, 0, 0));
  let c = hash3f(i + vec3i(0, 1, 0));
  let d = hash3f(i + vec3i(1, 1, 0));
  let e = hash3f(i + vec3i(0, 0, 1));
  let g = hash3f(i + vec3i(1, 0, 1));
  let h = hash3f(i + vec3i(0, 1, 1));
  let k = hash3f(i + vec3i(1, 1, 1));
  return mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y), mix(mix(e, g, u.x), mix(h, k, u.x), u.y), u.z);
}
fn fbm3(p: vec3f, oct: i32) -> f32 {
  var s = 0.0; var a = 0.5; var q = p; var norm = 0.0;
  for (var o = 0; o < oct; o++) {
    s += a * vnoise3(q);
    norm += a;
    q = q * 2.03 + vec3f(1.7, 9.2, 3.1);
    a *= 0.5;
  }
  return s / norm;
}
`;
