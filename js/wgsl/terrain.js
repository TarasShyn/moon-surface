import { FRAME_WGSL, LAYER_FUNCS, TEX, TEX_MIPS, SH_RES } from './common.js';

const BINDINGS = /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var hTex: texture_2d_array<f32>;
@group(0) @binding(2) var mTex: texture_2d_array<f32>;
@group(0) @binding(3) var sampR: sampler;
@group(0) @binding(4) var shTex: texture_2d_array<f32>;
@group(0) @binding(5) var sampC: sampler;
@group(0) @binding(6) var hmTex: texture_2d_array<f32>;
`;

const CONSTS = /* wgsl */ `
const TEXN: f32 = ${TEX}.0;
const MAXLOD: f32 = ${TEX_MIPS - 1}.0;
const HMAX: f32 = 4500.0;
`;

// Terrain height along a sun ray relative to its start x0. Bilinear weights are quantised to a
// few bits, so a layer whose texels are much larger than t is extrapolated from its gradient at
// x0 instead of sampled; otherwise its sampled difference would be mostly quantisation noise.
const MARCH_FUNCS = (uv) => /* wgsl */ `
struct Local { h0: array<f32, 5>, g: array<vec2f, 5>, y0: f32 };

fn layerLod(foot: f32, k: u32) -> f32 { return log2(max(foot * TEXN / layerPeriod(k), 1e-6)); }
fn layerMean(k: u32) -> f32 { return F.layerMean[k >> 2u][k & 3u]; }
fn layerHeight(p: vec2f, k: u32, lod: f32) -> f32 {
  if (lod > MAXLOD - 0.5) { return layerMean(k); }
  return textureSampleLevel(hmTex, sampR, ${uv}(p, k), k, max(lod, 0.0)).r;
}
fn relHeight(x0: vec2f, sh: vec2f, t: f32, foot: f32, L: Local) -> f32 {
  let p = x0 + sh * t;
  var dh = 0.0;
  for (var k = 0u; k < 5u; k++) {
    if (layerPeriod(k) > 2.0 * t * TEXN) {
      dh += dot(L.g[k], sh) * t;
    } else {
      dh += layerHeight(p, k, layerLod(foot, k)) - L.h0[k];
    }
  }
  return dh;
}
`;

export const TERRAIN = FRAME_WGSL + BINDINGS + LAYER_FUNCS + CONSTS + MARCH_FUNCS('layerUV') + /* wgsl */ `
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) xz: vec2f,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) inst: vec4f) -> VSOut {
  let n = u32(F.lodP.y);
  let gi = vec2f(f32(vi % (n + 1u)), f32(vi / (n + 1u)));
  let spacing = inst.z / f32(n);
  var xz = inst.xy + gi * spacing;
  let R = F.lodP.x * exp2(inst.w);
  let dy = max(max(F.lodP.z - F.camPos.y, 0.0), F.camPos.y - F.lodP.w);
  let d = length(vec3f(xz.x, dy, xz.y));
  let mk = clamp((d - 0.8 * R) / (0.2 * R), 0.0, 1.0);
  xz -= fract(gi * 0.5) * 2.0 * spacing * mk;
  let eff = spacing * (1.0 + mk);
  var h = 0.0;
  for (var k = 0u; k < 5u; k++) {
    let lod = clamp(log2(max(eff * TEXN / layerPeriod(k), 1.0)) + 0.5, 0.0, MAXLOD);
    h += textureSampleLevel(hmTex, sampR, layerUV(xz, k), k, lod).r;
  }
  var o: VSOut;
  o.rel = vec3f(xz.x, h - dot(xz, xz) / (2.0 * MOON_R) - F.camPos.y, xz.y);
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  o.xz = xz;
  return o;
}

// Horizon tangent beyond 6 texels of the finest shadow clipmap level that covers xz, corrected
// for the receiver's own height. Returns (tangent, distance the clipmap takes over from, level).
fn farHorizon(xz: vec2f, y0: f32) -> vec3f {
  for (var l = 0u; l < 5u; l++) {
    let L = F.shadowLvl[l];
    if (L.w < 0.0) { continue; }
    let d = xz - L.xy;
    if (max(abs(d.x), abs(d.y)) < 0.45 * L.z) {
      let s = textureSampleLevel(shTex, sampC, d / L.z + 0.5, u32(L.w), 0.0);
      return vec3f(s.x + (s.z - y0) / max(s.y, 1e-3), 6.0 * L.z / ${SH_RES}.0, f32(l));
    }
  }
  return vec3f(-1e3, 1e9, -1.0);
}

// Visible fraction of the solar disc. The short range is marched per pixel; tangents are in the
// receiver's local frame, where the surface drops by t^2/2R and the sun stands higher by the
// distance travelled toward it over R.
fn sunShadow(xz: vec2f, s0: f32, L: Local) -> f32 {
  let sun = F.sunDir.xyz;
  let hl = length(sun.xz);
  if (hl < 1e-4) { return 1.0; }
  let sh = sun.xz / hl;
  let tanE = sun.y / hl + dot(xz, sh) / MOON_R;
  let bias = 0.004 + s0 * 0.15;
  let far = farHorizon(xz, L.y0 + bias);
  var maxA = far.x;
  if (maxA > tanE + SUN_ANG_R) { return 0.0; }
  let steps = i32(F.shade.w);
  var t = s0;
  for (var i = 0; i < steps && t < far.y; i++) {
    let dh = relHeight(xz, sh, t, max(t * 0.6, s0), L) - bias - t * t / (2.0 * MOON_R);
    maxA = max(maxA, dh / t);
    if (maxA > tanE + SUN_ANG_R) { return 0.0; }
    t *= 1.6;
  }
  return smoothstep(-SUN_ANG_R, SUN_ANG_R, tanE - maxA);
}

// Lunar-Lambert (McEwen 1991) with an opposition surge: no limb darkening, strong backscatter.
fn lunarBRDF(mu0: f32, mu: f32, cosA: f32) -> f32 {
  let a = acos(clamp(cosA, -1.0, 1.0));
  let ad = a * 57.2958;
  let Lp = clamp(1.0 - 0.019 * ad + 2.42e-4 * ad * ad - 1.46e-6 * ad * ad * ad, 0.0, 1.0);
  let m0 = max(mu0, 0.0);
  let ls = 2.0 * Lp * m0 / (m0 + max(mu, 0.02)) + (1.0 - Lp) * m0;
  let opp = 1.0 + 0.8 / (1.0 + tan(min(a, 1.5) * 0.5) / 0.04);
  return ls * opp * exp(-0.45 * a);
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let xz = in.xz;
  var loc: Local;
  var g = vec2f(0.0);
  var alb = 1.0;
  var rock = 0.0;
  var mare = 0.0;
  for (var k = 0u; k < 6u; k++) {
    let uv = layerUV(xz, k);
    let t = textureSample(hTex, sampR, uv, k);
    let gk = gradToWorld(t.yz, k);
    g += gk;
    if (k < 5u) { loc.g[k] = gk; }
    let m = textureSample(mTex, sampR, uv, k);
    alb *= m.r * 2.0;
    rock = max(rock, m.g);
    mare = max(mare, m.a);
  }
  let nrm = normalize(vec3f(-g.x + xz.x / MOON_R, 1.0, -g.y + xz.y / MOON_R));
  let up = normalize(vec3f(xz.x / MOON_R, 1.0, xz.y / MOON_R));
  let dist = length(in.rel);
  let V = -in.rel / dist;
  let sun = F.sunDir.xyz;
  let pixAng = 2.0 * length(F.camUp.xyz) * F.screen.w;

  let s0 = max(0.012, dist * pixAng * 1.5);
  let mu0 = dot(nrm, sun);
  var shadow = 0.0;
  if (mu0 > 0.0) {
    loc.y0 = 0.0;
    for (var k = 0u; k < 5u; k++) {
      loc.h0[k] = layerHeight(xz, k, layerLod(s0, k));
      loc.y0 += loc.h0[k];
    }
    shadow = sunShadow(xz, s0, loc);
  }

  let tint = mix(vec3f(1.0, 0.955, 0.89), vec3f(0.92, 0.925, 0.95), mare);
  let A = F.shade.x * alb * (1.0 - 0.15 * rock);
  let direct = F.sunDir.w * shadow * lunarBRDF(mu0, dot(nrm, V), dot(sun, V));
  let sinE = max(dot(sun, up), 0.0);
  let bounce = F.sunDir.w * sinE * F.shade.x * (0.03 + 0.5 * (1.0 - dot(nrm, up))) * F.shade.y;
  let ed = F.earthDir.xyz;
  let earthLit = 0.5 + 0.5 * dot(sun, -ed);
  let earthE = 8e-5 * F.shade.z * earthLit * max(dot(nrm, ed), 0.0);
  let col = tint * A / PI * (direct + bounce + earthE);

  if (F.post.w > 0.5) {
    if (F.post.w < 1.5) { return vec4f(nrm * 0.5 + 0.5, 1.0) * 0.02; }
    if (F.post.w < 2.5) { return vec4f(vec3f(shadow), 1.0) * 0.02; }
    return vec4f(tint * A, 1.0) * 0.1;
  }
  return vec4f(col, 1.0);
}
`;

// Readback arrives a few frames late, so besides the height under the camera it reports the
// highest point in a disc the camera can reach meanwhile and along its current heading.
export const PROBE = FRAME_WGSL + /* wgsl */ `
struct ProbeU { ahead: vec2f, radius: f32, pad: f32 };
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var hmTex: texture_2d_array<f32>;
@group(0) @binding(2) var sampR: sampler;
@group(0) @binding(3) var<storage, read_write> outH: array<f32, 4>;
@group(0) @binding(4) var<uniform> U: ProbeU;
` + LAYER_FUNCS + /* wgsl */ `
fn heightAt(rel: vec2f) -> f32 {
  var h = 0.0;
  for (var k = 0u; k < 5u; k++) {
    h += textureSampleLevel(hmTex, sampR, layerUV(rel, k), k, 0.0).r;
  }
  return h - dot(rel, rel) / (2.0 * MOON_R);
}
@compute @workgroup_size(1)
fn main() {
  let centre = heightAt(vec2f(0.0));
  var top = centre;
  for (var i = 0; i < 12; i++) {
    let a = f32(i) * 0.5235988;
    let d = vec2f(cos(a), sin(a));
    top = max(top, heightAt(d * U.radius * 0.5));
    top = max(top, heightAt(d * U.radius));
  }
  for (var j = 1; j <= 6; j++) {
    top = max(top, heightAt(U.ahead * (f32(j) / 6.0)));
  }
  outH[0] = centre;
  outH[1] = top;
}
`;

// Shadow clipmap update: horizon tangent toward the sun azimuth from each texel centre, marched
// from 6 texels out to 60 km. Stores (tangent, distance of the steepest occluder, texel height).
export const SHADOW_MAP = (fmt) => FRAME_WGSL + /* wgsl */ `
struct Job { off: array<vec4f, 3>, geo: vec4f, sun: vec4f };
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> J: Job;
@group(0) @binding(2) var hmTex: texture_2d_array<f32>;
@group(0) @binding(3) var sampR: sampler;
@group(0) @binding(4) var outTex: texture_storage_2d<${fmt}, write>;
@group(0) @binding(5) var hTex: texture_2d_array<f32>;
const STEPS: i32 = 40;
` + LAYER_FUNCS + CONSTS + /* wgsl */ `
fn jobUV(pos: vec2f, k: u32) -> vec2f {
  let cs = layerCS(k);
  let v = J.off[k >> 1u];
  let o = select(v.xy, v.zw, (k & 1u) == 1u);
  return o + vec2f(cs.x * pos.x - cs.y * pos.y, cs.y * pos.x + cs.x * pos.y) / layerPeriod(k);
}
` + MARCH_FUNCS('jobUV') + /* wgsl */ `
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let ij = vec2u(gid.x, gid.y + u32(J.geo.w));
  if (ij.y >= ${SH_RES}u) { return; }
  let texel = J.geo.z;
  let pos = J.geo.xy + (vec2f(ij) + 0.5) * texel;
  let sh = J.sun.xy;
  var loc: Local;
  loc.y0 = 0.0;
  for (var k = 0u; k < 5u; k++) {
    let lod = layerLod(texel, k);
    loc.h0[k] = layerHeight(pos, k, lod);
    loc.y0 += loc.h0[k];
    let d = textureSampleLevel(hTex, sampR, jobUV(pos, k), k, clamp(lod, 0.0, MAXLOD));
    loc.g[k] = gradToWorld(d.yz, k);
  }
  let bias = 0.01 + 0.1 * texel;
  let y0 = loc.y0 + bias;
  let t0 = 6.0 * texel;
  let q = exp(log(60000.0 / t0) / f32(STEPS));
  var t = t0;
  var maxA = -1e3;
  var tMax = t0;
  for (var i = 0; i < STEPS; i++) {
    let drop = t * t / (2.0 * MOON_R);
    if (HMAX - drop - y0 < maxA * t) { break; }
    let a = (relHeight(pos, sh, t, t * (q - 1.0), loc) - bias - drop) / t;
    if (a > maxA) { maxA = a; tMax = t; }
    t *= q;
  }
  textureStore(outTex, ij, vec4f(maxA, tMax, y0, 1.0));
}
`;
