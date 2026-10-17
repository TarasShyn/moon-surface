# moon-surface

Live: https://tarasshynkarenko.com/moon

A real-time lunar surface renderer on WebGPU.

This project exists because of [web-ocean](https://github.com/Noniv/web-ocean) by Maksymilian Dendura ([@Noniv](https://github.com/Noniv)). Go see the ocean: https://ocean-noniv.vercel.app. I took its approach and pointed it at the Moon. That means no engine, a CDLOD mesh over summed tiled detail layers, a free camera that stays above the surface, camera-centred world-space clipmaps, and HDR with bloom and AgX tone mapping. The bloom and tone mapping code in `js/wgsl/post.js` is his, adapted under the MIT license.

## Run it

```
npm start        # http://localhost:8090/
npm run build    # moon.html, one self-contained page that also opens from file://
```

Needs a recent Chrome with WebGPU. Heights use `float32-filterable` when the GPU has it; without it they fall back to fp16 and show 1 m contour steps.

## Controls

| Input | Action |
|---|---|
| Click | Capture the mouse (Esc releases). Right-drag also looks around. |
| W A S D | Move |
| Space or E / Q or C | Up / down |
| Shift | 5x speed |
| Wheel | Base flying speed |
| G | Walk mode under lunar gravity (Space jumps) |
| Z | Zoom to 6 degrees, for looking at the Earth |
| H | Settings: sun, Earth, exposure, sky time lapse, sun follows sky, albedo, bounce light, earthshine, stars, shadow steps, resolution |

URL parameters: `cam=x,altitude,z,yawDeg,pitchDeg`, `fov=deg`, `scale=0.5`, `ui=0`, `dbg=1|2|3` (normals, shadow, albedo), and any settings key, for example `sunElev=4&sunAzim=170`.

## How it works

**Terrain.** Six 1024x1024 layers are generated once on the GPU, with periods from 131 km down to 3.3 m, each rotated and offset so the tiling never lines up. Each layer has three octaves of craters whose size, depth, rim, flat floor, terraces and central peak follow the crater's radius and age. Young craters get bright ejecta and rays. The 131 km layer also holds ridged highlands, darker maria, and a 48 km complex crater you start inside. The finer layers carry three octaves of rocks, sized to stay 3 to 30 texels wide, denser on fresh crater rims, and rougher fbm regolith. All sampling is camera-relative, so precision holds anywhere.

**Shape of the world.** The Moon is 1737.4 km in radius, so the surface drops by d²/2R. From eye height the horizon is about 2.4 km away, and crater walls 30 km off sink a few hundred metres.

**Shading.** Lunar-Lambert reflectance (McEwen) with an opposition surge, so there is no limb darkening and the down-sun view goes flat and bright. Fill light comes from sunlit terrain bouncing into shadows and from earthshine, which scales with the Earth's phase. Lower the sun below the horizon and raise exposure to about 14 EV to see the surface lit by the Earth alone.

**Shadows.** Shadows are soft: the shading is the fraction of the 0.53° solar disc that is visible. Long-range horizons live in five camera-centred world-space clipmaps (64 m to 262 km), rebuilt a few rows per frame when the camera moves or the sun azimuth changes. Each pixel marches only the short range itself. Layers whose texels are much larger than the march step are extrapolated from their gradient rather than sampled, which keeps bilinear quantisation from turning into shadow acne.

**Moving sky.** The stars and Milky Way turn about the lunar pole at the synodic rate (one turn per 29.5 days) times the time lapse, 4000x by default, so they drift about 0.5 degrees a second. The Earth spins once per sidereal day at the same lapse, and its clouds drift. The Earth itself stays put, as it does from the Moon's near side. With "Sun follows sky" set to 1 the sun moves along the same path, and shadows follow it.

**Staying above ground.** The height probe reports the highest point in a disc the camera can reach before the next readback, plus points along its heading. The camera is hard-clamped 1 m above that, so diving or flying into a slope never puts it under the surface.

**Sky.** Black, with procedural stars, a faint Milky Way, the Sun disc with a tight lens glare, and the Earth at 1.9 degrees: continents, clouds, ice caps, an ocean glint, a thin atmosphere limb, city lights on the night side, and its phase from the sun direction.

## Credits

- [web-ocean](https://github.com/Noniv/web-ocean) by Maksymilian Dendura: the architecture this is built on, and the bloom and AgX tone mapping in `js/wgsl/post.js` (MIT).
- McEwen (1991), the lunar-Lambert reflectance model used for the regolith.
- Built with Claude Code (Opus 5.5).

## License

MIT. See [LICENSE](LICENSE). The web-ocean copyright notice for `js/wgsl/post.js` is kept there as its license requires.
