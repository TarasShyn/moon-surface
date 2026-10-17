// Procedural terrain layers, generated once at start-up. Each layer is a tileable heightfield
// whose features scale with its period, so summing the six layers covers craters from
// 25 km down to centimetre pebbles.
import { NOISE_WGSL, TEX, LAYER_P, HERO_R, wgslF } from './common.js';

const LAYER_UNIFORM = /* wgsl */ `
struct GenU { layer: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> U: GenU;
const TEXF: f32 = ${TEX}.0;
var<private> PER: array<f32, 6> = array<f32, 6>(${LAYER_P.map(wgslF).join(', ')});
`;

export const GEN = NOISE_WGSL + LAYER_UNIFORM + /* wgsl */ `
@group(0) @binding(1) var hOut: texture_storage_2d_array<r32float, write>;
@group(0) @binding(2) var mOut: texture_storage_2d_array<rgba8unorm, write>;

const HERO_R: f32 = ${wgslF(HERO_R)};
var<private> RELIEF: array<f32, 6> = array<f32, 6>(0.0040, 0.0030, 0.0026, 0.0022, 0.0024, 0.0040);
var<private> ROUGH_GAIN: array<f32, 6> = array<f32, 6>(0.5, 0.5, 0.52, 0.56, 0.6, 0.64);
var<private> BOULDER_D: array<f32, 6> = array<f32, 6>(0.0, 0.0, 0.0, 0.35, 0.6, 0.8);
var<private> CRATER_D: array<f32, 6> = array<f32, 6>(1.0, 1.0, 0.9, 0.7, 0.45, 0.25);
var<private> AGE_MIN: array<f32, 6> = array<f32, 6>(0.0, 0.0, 0.1, 0.3, 0.45, 0.6);

struct CF { h: f32, fresh: f32, rim: f32 };

fn wrapDelta(d: vec2f) -> vec2f { return d - round(d); }

// d: offset from the crater centre in crater radii. Returns (height m, fresh ejecta, rim weight).
fn craterProfile(d: vec2f, Rm: f32, age: f32, sf: f32) -> vec3f {
  let l = length(d);
  let dirv = d / max(l, 1e-5);
  let complex = smoothstep(2500.0, 9000.0, Rm);
  let irr = gnoise(dirv * 2.2 + sf) * 0.6 + gnoise(dirv * 5.3 + sf * 1.3) * 0.3 + gnoise(d * 3.0 + sf * 0.7) * 0.25;
  let r = l * (1.0 + (0.06 + 0.05 * complex) * irr);
  let depth = mix(0.40, 0.165, complex) * (1.0 - 0.62 * age) * Rm;
  let rimRel = mix(0.22, 0.30, complex) * (1.0 - 0.75 * age);
  let pw = mix(2.0, 3.6, complex);
  let wr = r + complex * 0.035 * sin(r * 31.0 + irr * 2.0) * smoothstep(0.5, 0.95, r);
  let inner = pow(clamp(wr, 0.0, 1.8), pw) - 1.0 + rimRel;
  let outer = rimRel * pow(max(r, 0.6), -3.0) * (1.0 - smoothstep(1.7, 2.6, r));
  var s = smin(inner, outer, 0.04 + 0.25 * age);
  let floorL = -1.0 + rimRel + 0.32 * complex + 0.3 * age;
  s = smax(s, floorL, 0.06 + 0.2 * age);
  s += complex * 0.42 * exp(-l * l / 0.035) * (0.55 + 0.9 * abs(gnoise(d * 6.0 + sf)));
  let youth = 1.0 - smoothstep(0.02, 0.2, age);
  let rays = smoothstep(0.15, 0.55, gnoise(dirv * 9.0 + sf * 2.1) + 0.5 * gnoise(dirv * 23.0 + sf))
    * (1.0 - smoothstep(1.0, 2.6, r)) * smoothstep(0.9, 1.1, r);
  let halo = 1.0 - smoothstep(0.6, 1.7, r);
  let fresh = youth * max(halo, rays * 0.8);
  let rim = exp(-(r - 1.0) * (r - 1.0) / 0.06) * (1.0 - age) + halo * youth * 0.5;
  return vec3f(s * depth, fresh, rim);
}

// Crater centres stay 0.15 cells from their cell edges and effects end at 2.6 radii, so a
// 3x3 neighbourhood is enough when the radius is at most 0.4 cells.
fn craterOctave(p: vec2f, n: i32, seed: u32, P: f32, dens: f32, ageMin: f32, cf: ptr<function, CF>) {
  let nf = f32(n);
  let q = p * nf;
  let c = vec2i(floor(q));
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let cc = c + vec2i(i, j);
      let cw = ((cc % n) + n) % n;
      let r4 = hash4(cw, seed);
      if (r4.x > dens) { continue; }
      let ctr = vec2f(cc) + 0.15 + 0.7 * r4.yz;
      let Rc = mix(0.12, 0.40, pow(r4.w, 2.2));
      let d = (q - ctr) / Rc;
      if (dot(d, d) > 6.76) { continue; }
      let r4b = hash4(cw, seed ^ 0x9e3779b9u);
      let res = craterProfile(d, Rc / nf * P, mix(ageMin, 1.0, sqrt(r4b.x)), r4b.y * 100.0);
      (*cf).h += res.x;
      (*cf).fresh = max((*cf).fresh, res.y);
      (*cf).rim = max((*cf).rim, res.z);
    }
  }
}

// Rock radii span rMin..rMax cells, so with n cells per tile they stay 3-30 texels wide.
fn boulderOctave(p: vec2f, n: i32, seed: u32, P: f32, dens: f32, rMin: f32, rMax: f32) -> vec2f {
  let nf = f32(n);
  let q = p * nf;
  let c = vec2i(floor(q));
  var hb = 0.0;
  var rk = 0.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let cc = c + vec2i(i, j);
      let cw = ((cc % n) + n) % n;
      let r = hash4(cw, seed);
      if (r.x > dens + 0.06) { continue; }
      let present = smoothstep(r.x, r.x + 0.06, dens);
      if (present <= 0.0) { continue; }
      let r2 = hash4(cw, seed ^ 0x68bc21ebu);
      let ctr = vec2f(cc) + 0.3 + 0.4 * r.yz;
      let Rc = mix(rMin, rMax, pow(r.w, 2.5));
      let Rm = Rc / nf * P;
      var lv = (q - ctr) / Rc;
      let a = r2.x * 6.2831853;
      let cs = cos(a);
      let sn = sin(a);
      lv = vec2f(cs * lv.x - sn * lv.y, sn * lv.x + cs * lv.y) * vec2f(1.0, mix(1.0, 1.6, r2.y));
      let l2 = dot(lv, lv);
      if (l2 > 2.0) { continue; }
      let sf = r2.z * 50.0;
      let bump = gnoise(lv * 1.4 + sf) * 0.28 + gnoise(lv * 3.3 + sf) * 0.12 + gnoise(lv * 7.5 + sf) * 0.05;
      let box = max(abs(lv.x), abs(lv.y));
      let qd = mix(sqrt(l2), box * 1.05, 0.35) * (1.0 + bump);
      let prof = sqrt(max(1.0 - qd * qd, 0.0));
      let b = Rm * (mix(0.55, 0.9, r2.w) * prof - 0.12 - 0.15 * r.y) * present;
      hb = max(hb, b);
      rk = max(rk, smoothstep(0.0, 0.06 * Rm, b));
    }
  }
  return vec2f(hb, rk);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = U.layer;
  let P = PER[k];
  let fk = f32(k);
  let p = (vec2f(gid.xy) + 0.5) / TEXF;
  let seed = k * 7919u + 101u;

  let relief = pfbmGain(p * 4.0 + vec2f(fk * 17.3, fk * 5.1), 4, 8, ROUGH_GAIN[k]);
  var h = relief * RELIEF[k] * P;

  var mare = 0.0;
  var heroR = 99.0;
  var heroD = vec2f(0.0);
  var highland = 1.0;
  if (k == 0u) {
    heroD = wrapDelta(p - vec2f(0.5)) * P / HERO_R;
    heroR = length(heroD);
    let ridge = pridged(p * 6.0 + vec2f(3.3, 1.7), 6, 6);
    highland = smoothstep(-0.12, 0.22, pfbm(p * 3.0 + vec2f(0.7, 8.1), 3, 3));
    mare = smoothstep(0.04, 0.2, pfbm(p * 2.0 + vec2f(9.1, 2.3), 2, 4)) * smoothstep(1.4, 2.4, heroR);
    let hl = h + ridge * 1400.0 * (0.35 + 0.65 * highland);
    h = mix(hl, -900.0 + 0.2 * h, mare);
  }

  var cf: CF;
  cf.h = 0.0;
  cf.fresh = 0.0;
  cf.rim = 0.0;
  let cd = CRATER_D[k];
  let am = AGE_MIN[k];
  craterOctave(p, 4, seed + 1u, P, 0.55 * cd, am, &cf);
  craterOctave(p, 8, seed + 2u, P, 0.70 * cd, am, &cf);
  craterOctave(p, 16, seed + 3u, P, 0.85 * cd, am, &cf);
  var craterW = 1.0 - 0.6 * mare;
  if (k == 0u) {
    craterW *= smoothstep(0.8, 1.15, heroR);
    cf.fresh *= smoothstep(0.8, 1.15, heroR);
    let hero = craterProfile(heroD, HERO_R, 0.04, 7.0);
    h += hero.x * 1.45;
    cf.fresh = max(cf.fresh, hero.y);
  }
  h += cf.h * craterW;

  var rock = 0.0;
  if (BOULDER_D[k] > 0.0) {
    let dens = clamp(BOULDER_D[k] * (0.5 + 3.0 * cf.rim + 0.8 * cf.fresh), 0.0, 1.0);
    let b0 = boulderOctave(p, 10, seed + 9u, P, dens * 0.3, 0.12, 0.3);
    let b1 = boulderOctave(p, 20, seed + 10u, P, dens * 0.45, 0.12, 0.3);
    let b2 = boulderOctave(p, 40, seed + 11u, P, dens * 0.65, 0.12, 0.3);
    h += max(max(b0.x, b1.x), b2.x);
    rock = max(max(b0.y, b1.y), b2.y);
  }

  let albN = pfbm(p * 8.0 + vec2f(fk * 3.7, 1.3), 8, 4);
  var alb = (1.0 + 0.36 * albN) * (1.0 + 0.9 * cf.fresh) * (1.0 - 0.45 * mare);
  if (k == 0u) { alb *= mix(0.92, 1.08, highland); }
  textureStore(hOut, gid.xy, k, vec4f(h, 0.0, 0.0, 0.0));
  textureStore(mOut, gid.xy, k, vec4f(clamp(alb * 0.5, 0.0, 1.0), rock, cf.fresh, mare));
}
`;

// x + w is the height (split into an f16 part and its residual so the rgba16float fallback keeps
// precision when sampled unfiltered); y and z are the slope along the layer's local axes.
export const DERIV = (fmt, hfmt) => LAYER_UNIFORM + /* wgsl */ `
@group(0) @binding(1) var hIn: texture_2d_array<f32>;
@group(0) @binding(2) var hOut: texture_storage_2d_array<${fmt}, write>;
@group(0) @binding(3) var hOnly: texture_storage_2d_array<${hfmt}, write>;
fn ld(c: vec2i, k: u32) -> f32 {
  let n = i32(TEXF);
  return textureLoad(hIn, (c + n) % n, k, 0).r;
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = U.layer;
  let c = vec2i(gid.xy);
  let texel = PER[k] / TEXF;
  let h = ld(c, k);
  let hx = (ld(c + vec2i(1, 0), k) - ld(c - vec2i(1, 0), k)) / (2.0 * texel);
  let hz = (ld(c + vec2i(0, 1), k) - ld(c - vec2i(0, 1), k)) / (2.0 * texel);
  let hi = quantizeToF16(h);
  textureStore(hOut, gid.xy, k, vec4f(hi, hx, hz, h - hi));
  textureStore(hOnly, gid.xy, k, vec4f(h, 0.0, 0.0, 0.0));
}
`;

export const HEIGHT_MIPS = (fmt) => /* wgsl */ `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<${fmt}, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dst);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let c = vec2i(gid.xy) * 2;
  let k = gid.z;
  let a = textureLoad(src, c, k, 0);
  let b = textureLoad(src, c + vec2i(1, 0), k, 0);
  let d = textureLoad(src, c + vec2i(0, 1), k, 0);
  let e = textureLoad(src, c + vec2i(1, 1), k, 0);
  let s = (a + b + d + e) * 0.25;
  let h = s.x + s.w;
  let hi = quantizeToF16(h);
  textureStore(dst, gid.xy, k, vec4f(hi, s.y, s.z, h - hi));
}
`;

export const SCALAR_MIPS = (fmt) => /* wgsl */ `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<${fmt}, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dst);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let c = vec2i(gid.xy) * 2;
  let k = gid.z;
  let s = textureLoad(src, c, k, 0).r + textureLoad(src, c + vec2i(1, 0), k, 0).r
        + textureLoad(src, c + vec2i(0, 1), k, 0).r + textureLoad(src, c + vec2i(1, 1), k, 0).r;
  textureStore(dst, gid.xy, k, vec4f(s * 0.25, 0.0, 0.0, 0.0));
}
`;

export const MAT_MIPS = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba8unorm, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dst);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let c = vec2i(gid.xy) * 2;
  let k = gid.z;
  let s = textureLoad(src, c, k, 0) + textureLoad(src, c + vec2i(1, 0), k, 0)
        + textureLoad(src, c + vec2i(0, 1), k, 0) + textureLoad(src, c + vec2i(1, 1), k, 0);
  textureStore(dst, gid.xy, k, s * 0.25);
}
`;
