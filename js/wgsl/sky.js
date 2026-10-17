// Black airless sky: stars, Milky Way, the Sun with lens glare, and the Earth.
import { FRAME_WGSL, NOISE_WGSL } from './common.js';

export const SKY = FRAME_WGSL + NOISE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;

struct VO { @builtin(position) pos: vec4f, @location(0) ndc: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VO;
  o.pos = vec4f(p, 0.0, 1.0);
  o.ndc = p;
  return o;
}

fn starDir(face: u32, sp: vec2f) -> vec3f {
  switch face {
    case 0u: { return normalize(vec3f(1.0, sp.x, sp.y)); }
    case 1u: { return normalize(vec3f(-1.0, sp.x, sp.y)); }
    case 2u: { return normalize(vec3f(sp.x, 1.0, sp.y)); }
    case 3u: { return normalize(vec3f(sp.x, -1.0, sp.y)); }
    case 4u: { return normalize(vec3f(sp.x, sp.y, 1.0)); }
    default: { return normalize(vec3f(sp.x, sp.y, -1.0)); }
  }
}

// One star per cube-map cell, kept 0.25 cells from the edges so no neighbour lookups are needed.
fn starLayer(dir: vec3f, N: f32, seed: u32, prob: f32, fmin: f32, fmax: f32, sigma: f32) -> vec3f {
  let a = abs(dir);
  var face = 0u;
  var uv = vec2f(0.0);
  if (a.x >= a.y && a.x >= a.z) { face = select(1u, 0u, dir.x > 0.0); uv = dir.yz / a.x; }
  else if (a.y >= a.z) { face = select(3u, 2u, dir.y > 0.0); uv = dir.xz / a.y; }
  else { face = select(5u, 4u, dir.z > 0.0); uv = dir.xy / a.z; }
  let cell = vec2i(floor((uv * 0.5 + 0.5) * N));
  let h = hash4(cell + vec2i(i32(face) * 4099, i32(face) * 131), seed);
  if (h.x > prob) { return vec3f(0.0); }
  let sp = (vec2f(cell) + 0.25 + 0.5 * h.yz) / N * 2.0 - 1.0;
  let d = length(dir - starDir(face, sp));
  let flux = mix(fmin, fmax, pow(h.w, 5.0));
  let temp = hash4(cell, seed ^ 0xabcdefu).x;
  let col = select(mix(vec3f(1.0, 0.97, 0.92), vec3f(0.72, 0.82, 1.0), (temp - 0.6) / 0.4),
                   mix(vec3f(1.0, 0.62, 0.38), vec3f(1.0, 0.97, 0.92), temp / 0.6), temp < 0.6);
  return col * flux * exp(-d * d / (2.0 * sigma * sigma)) / (6.2832 * sigma * sigma);
}

fn galacticBand(dir: vec3f) -> f32 {
  let b = dot(dir, normalize(vec3f(-0.42, 0.62, 0.66)));
  return exp(-b * b / 0.022);
}

fn stars(dir: vec3f, pixAng: f32) -> vec3f {
  let sigma = max(pixAng * 0.6, 2e-5);
  let band = galacticBand(dir);
  var s = starLayer(dir, 40.0, 11u, 0.4, 4e-9, 9e-8, sigma);
  s += starLayer(dir, 110.0, 23u, 0.45, 8e-10, 6e-9, sigma);
  s += starLayer(dir, 300.0, 37u, 0.25 + 0.6 * band, 3e-10, 1.2e-9, sigma);
  return s;
}

fn milkyWay(dir: vec3f) -> vec3f {
  let gN = normalize(vec3f(-0.42, 0.62, 0.66));
  let gC = normalize(cross(gN, vec3f(1.0, 0.0, 0.0)));
  let b = dot(dir, gN);
  let band = exp(-b * b / 0.022);
  let bulge = exp(-(1.0 - dot(dir, gC)) * 5.0) * exp(-b * b / 0.05);
  let s = fbm3(dir * 7.0 + 3.0, 5);
  let dust = smoothstep(0.48, 0.66, fbm3(dir * 11.0 + 17.0, 5)) * exp(-b * b / 0.004);
  let I = (band * (0.35 + 0.9 * s * s) + bulge * 1.4) * (1.0 - 0.85 * dust);
  return vec3f(1.0, 0.93, 0.82) * I * 4e-4;
}

fn sunDisk(dir: vec3f, pixAng: f32) -> vec3f {
  let sun = F.sunDir.xyz;
  let th = length(dir - sun);
  let w = max(pixAng, 1e-5);
  let cov = 1.0 - smoothstep(SUN_ANG_R - w, SUN_ANG_R + w, th);
  let x = clamp(th / SUN_ANG_R, 0.0, 1.0);
  let limb = 1.0 - 0.6 * (1.0 - sqrt(1.0 - x * x));
  let disk = cov * limb / (PI * SUN_ANG_R * SUN_ANG_R);
  let glare = 0.02 * exp(-th / 0.008) + 3e-4 * exp(-th / 0.06);
  return vec3f(1.0, 0.985, 0.96) * F.sunDir.w * (disk + glare * smoothstep(-0.01, 0.005, sun.y));
}

fn earth(dir: vec3f, pixAng: f32) -> vec4f {
  let ed = F.earthDir.xyz;
  let rE = F.earthDir.w;
  if (dot(dir, ed) < cos(rE * 1.15)) { return vec4f(0.0); }
  let sun = F.sunDir.xyz;
  let eu = normalize(cross(ed, vec3f(0.0, 1.0, 0.0)));
  let ev = cross(eu, ed);
  let xy = vec2f(dot(dir, eu), dot(dir, ev)) / rE;
  let rho = length(xy);
  let aa = pixAng / rE;
  let cover = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, rho);
  let rimN = normalize(eu * xy.x + ev * xy.y);
  let halo = exp(-max(rho - 1.0, 0.0) / 0.008) * smoothstep(-0.2, 0.35, dot(rimN, sun)) * 0.03;
  let haloCol = vec3f(0.30, 0.52, 1.0) * halo * (1.0 - cover) * F.earthAxis.w;
  if (cover <= 0.0) { return vec4f(haloCol, 0.0); }

  let xc = xy / max(rho, 1.0);
  let z = sqrt(max(1.0 - dot(xc, xc), 0.0));
  let nrm = normalize(eu * xc.x + ev * xc.y - ed * z);
  let ax = F.earthAxis.xyz;
  let b1 = normalize(cross(ax, vec3f(0.0, 0.0, 1.0)));
  let b2 = cross(ax, b1);
  let lat = asin(clamp(dot(nrm, ax), -1.0, 1.0));
  let lon = atan2(dot(nrm, b2), dot(nrm, b1)) + F.sky.z;
  let p3 = vec3f(cos(lat) * cos(lon), sin(lat), cos(lat) * sin(lon));
  let alat = abs(lat);

  let cn = fbm3(p3 * 1.7 + vec3f(4.1, 1.3, 7.7), 7);
  let land = smoothstep(0.54, 0.56, cn);
  let veg = fbm3(p3 * 3.3 + 11.0, 4);
  let desert = smoothstep(0.5, 0.58, veg) * (1.0 - smoothstep(0.2, 0.5, abs(alat - 0.42)));
  var surf = mix(vec3f(0.07, 0.10, 0.045), vec3f(0.38, 0.30, 0.19), desert);
  surf = mix(surf, vec3f(0.18, 0.16, 0.13), smoothstep(0.55, 0.7, fbm3(p3 * 7.0 + 3.0, 4)) * 0.5);
  let ice = smoothstep(1.12, 1.2, alat + 0.1 * (veg - 0.5));
  var alb = mix(vec3f(0.006, 0.016, 0.045), surf, land);
  alb = mix(alb, vec3f(0.75, 0.78, 0.82), max(ice, land * smoothstep(1.0, 1.1, alat)));

  let warp = vec3f(fbm3(p3 * 2.0 + 1.0, 3), fbm3(p3 * 2.0 + 5.0, 3), fbm3(p3 * 2.0 + 9.0, 3)) - 0.5;
  var cl = fbm3(p3 * 3.2 + warp * 1.8 + vec3f(F.sky.w * 0.02, 0.0, 0.0), 6);
  cl += 0.1 * exp(-lat * lat / 0.01) + 0.08 * exp(-(alat - 0.95) * (alat - 0.95) / 0.02);
  let cloud = smoothstep(0.55, 0.7, cl) * 0.9;
  alb = mix(alb, vec3f(0.85, 0.86, 0.88), cloud);

  let ndl = dot(nrm, sun);
  let day = smoothstep(-0.03, 0.15, ndl);
  var L = alb / PI * max(ndl, 0.0) + alb * 0.01 * smoothstep(-0.12, 0.05, ndl);
  let H = normalize(sun - dir);
  let water = (1.0 - land) * (1.0 - cloud) * (1.0 - ice);
  L += vec3f(1.0, 0.95, 0.85) * water * pow(max(dot(nrm, H), 0.0), 2500.0) * 3.0 * step(0.0, ndl);
  L += vec3f(0.28, 0.48, 1.0) * 0.03 * pow(1.0 - z, 3.0) * smoothstep(-0.15, 0.3, ndl);
  L += vec3f(0.006, 0.014, 0.036) * max(ndl, 0.0) * 0.3;
  let city = land * (1.0 - ice) * (1.0 - day) * smoothstep(0.66, 0.8, fbm3(p3 * 40.0, 3)) * (1.0 - cloud);
  L += vec3f(1.0, 0.7, 0.4) * city * 2e-4;
  return vec4f(L * F.sunDir.w * F.earthAxis.w * cover + haloCol, cover);
}

@fragment fn fs(in: VO) -> @location(0) vec4f {
  let dir = normalize(F.camFwd.xyz + in.ndc.x * F.camRight.xyz + in.ndc.y * F.camUp.xyz);
  let pixAng = 2.0 * length(F.camUp.xyz) * F.screen.w;
  let celestial = F.skyRot * dir;
  var L = milkyWay(celestial) * F.sky.y + stars(celestial, pixAng) * F.sky.x;
  let e = earth(dir, pixAng);
  L = L * (1.0 - e.a) + e.rgb;
  L += sunDisk(dir, pixAng);
  return vec4f(L, 1.0);
}
`;
