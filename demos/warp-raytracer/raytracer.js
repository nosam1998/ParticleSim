/*
 * Light rays through an Alcubierre bubble: the physics behind
 * demos/warp-raytracer/index.html, and nothing else.
 *
 * No DOM, no canvas: one export. The page claims to show what a passenger
 * inside the bubble sees, as particlesim.analysis.raytrace computes it, and
 * tests/unit/test_raytrace.py checks that claim by loading this file in Node.
 * It compares the rays traced here, and the colours painted from them, with
 * the Python module's, ray for ray.
 *
 * Two implementations of one algorithm:
 *
 *   traceRay  (JavaScript, double precision): the reference, and the page's
 *             fallback where WebGPU is missing.
 *   WGSL      (a WebGPU compute shader, single precision): one invocation
 *             per pixel, the same steps in the same order.
 *
 * The metric is ds^2 = -dt^2 + (dx - v f(r_s) dt)^2 + dy^2 + dz^2 with the
 * bubble centre at x = v t. Null geodesics are integrated in Hamiltonian
 * form, where with W = p_t + v f p_x:
 *
 *   dt/dl = -W,  dx/dl = p_x - v f W,  dy/dl = p_y,  dz/dl = p_z,
 *   dp_a/dl = v W p_x f'(r_s) d_a r_s
 *
 * by fourth-order Runge-Kutta, backwards from the passenger at the centre.
 * Each ray's step moves it at most `step` relative to the bubble in the wall,
 * growing as exp(0.4 sigma |r - R|) away from it, up to 2. A ray is done
 * where f vanishes (r > R + 20/sigma). It is trapped when its energy passes
 * 1e6, the sign of a horizon, where no light from the sky reaches the pixel.
 *
 * What does not depend on this code: p_t + v p_x is conserved, so each pixel's
 * light arrives shifted by exactly 1 - v cos(alpha). The page colours by it.
 */
(function (root) {
  "use strict";

  var BLUESHIFT = 1e6;

  function shape(r, b) {
    var s = b.sigma, R = b.R;
    return (Math.tanh(s * (r + R)) - Math.tanh(s * (r - R))) / (2 * Math.tanh(s * R));
  }

  function slope(r, b) {
    var s = b.sigma, R = b.R;
    var plus = Math.tanh(s * (r + R)), minus = Math.tanh(s * (r - R));
    return (s * ((1 - plus * plus) - (1 - minus * minus))) / (2 * Math.tanh(s * R));
  }

  function sky(b) {
    return b.R + 20 / b.sigma;
  }

  /* Hamilton's equations for one ray; s = [t, x, y, z, p_t, p_x, p_y, p_z]. */
  function rates(b, s) {
    var v = b.v;
    var xs = s[1] - v * s[0];
    var r = Math.sqrt(xs * xs + s[2] * s[2] + s[3] * s[3]);
    var f = shape(r, b);
    var overR = r > 0 ? slope(r, b) / r : 0;
    var W = s[4] + v * f * s[5];
    var force = v * W * s[5] * overR;
    return [-W, s[5] - v * f * W, s[6], s[7], force * (-v * xs), force * xs, force * s[2], force * s[3]];
  }

  function add(s, h, k) {
    var out = new Array(8);
    for (var i = 0; i < 8; i++) out[i] = s[i] + h * k[i];
    return out;
  }

  /* Trace the light a passenger at the centre sees from direction d (unit 3-vector). */
  function traceRay(b, d, step, maxSteps) {
    step = step || 0.01;
    maxSteps = maxSteps || 200000;
    var norm = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    var dx = d[0] / norm, dy = d[1] / norm, dz = d[2] / norm;
    var f0 = shape(0, b);
    var s = [0, 0, 0, 0, 1 - b.v * f0 * dx, dx, dy, dz];
    var edge = sky(b);
    for (var n = 1; n <= maxSteps; n++) {
      var k1 = rates(b, s);
      var xs = s[1] - b.v * s[0];
      var r = Math.sqrt(xs * xs + s[2] * s[2] + s[3] * s[3]);
      var rel = k1[1] - b.v * k1[0];
      var speed = Math.sqrt(rel * rel + k1[2] * k1[2] + k1[3] * k1[3]);
      var reach = Math.min(step * Math.exp(0.4 * b.sigma * Math.abs(r - b.R)), 2);
      var h = reach / Math.max(speed, 1e-300);
      var k2 = rates(b, add(s, 0.5 * h, k1));
      var k3 = rates(b, add(s, 0.5 * h, k2));
      var k4 = rates(b, add(s, h, k3));
      for (var i = 0; i < 8; i++) s[i] += (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
      xs = s[1] - b.v * s[0];
      r = Math.sqrt(xs * xs + s[2] * s[2] + s[3] * s[3]);
      var lost = Math.abs(s[4]) > BLUESHIFT;
      if (lost) return { trapped: true, sky: null, doppler: NaN, steps: n };
      if (r > edge) {
        var p = Math.sqrt(s[5] * s[5] + s[6] * s[6] + s[7] * s[7]);
        return { trapped: false, sky: [s[5] / p, s[6] / p, s[7] / p], doppler: 1 / s[4], steps: n };
      }
    }
    return { trapped: true, sky: null, doppler: NaN, steps: maxSteps };
  }

  /* Equirectangular: longitude from +x across the width, straight ahead in the middle. */
  function panoramaDirection(i, j, width, height) {
    var lon = ((i + 0.5) / width) * 2 * Math.PI - Math.PI;
    var lat = Math.PI / 2 - ((j + 0.5) / height) * Math.PI;
    return [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
  }

  function mod2(a) {
    return ((a % 2) + 2) % 2;
  }

  /* The reference sky: checkerboard, warm ahead, cool behind, grid every 30 degrees. */
  function skyColour(d) {
    var lon = Math.atan2(d[1], d[0]);
    var lat = Math.asin(Math.max(-1, Math.min(1, d[2])));
    var cell = Math.PI / 12;
    var checker = mod2(Math.floor(lon / cell) + Math.floor(lat / cell));
    var ahead = 0.5 * (1 + Math.cos(lon) * Math.cos(lat));
    var warm = [0.95, 0.55, 0.25], cool = [0.25, 0.55, 0.95];
    var shade = 0.55 + 0.45 * checker;
    var rgb = [0, 1, 2].map(function (c) { return (ahead * warm[c] + (1 - ahead) * cool[c]) * shade; });
    var step = Math.PI / 6;
    var meridian = Math.abs(lon / step - Math.round(lon / step)) * Math.cos(lat);
    if (Math.abs(lat) > (80 * Math.PI) / 180) meridian = 1;
    var near = Math.min(meridian, Math.abs(lat / step - Math.round(lat / step)));
    if (near < 0.02) rgb = [0.95, 0.95, 0.95];
    return rgb;
  }

  /* A pixel's colour: the sky it sees, tinted and brightened by the Doppler factor. */
  function shade(result, doppler) {
    if (result.trapped) return [0, 0, 0];
    var rgb = skyColour(result.sky);
    if (doppler !== false) {
      var D = result.doppler;
      var shift = Math.tanh(Math.log(D));
      var tint = shift > 0 ? [0.35, 0.55, 1.0] : [1.0, 0.35, 0.25];
      var a = Math.abs(shift);
      var gain = Math.pow(Math.min(Math.max(Math.pow(D, 4), 0.15), 3.0), 0.25);
      rgb = rgb.map(function (c, k) { return ((1 - 0.5 * a) * c + 0.5 * a * tint[k]) * gain; });
    }
    return rgb.map(function (c) { return Math.max(0, Math.min(1, c)); });
  }

  function toByte(c) {
    return Math.floor(c * 255 + 0.5);
  }

  /* The whole panorama on the CPU: RGBA bytes, row by row. */
  function renderCPU(b, width, height, step) {
    var out = new Uint8ClampedArray(width * height * 4);
    for (var j = 0; j < height; j++) {
      for (var i = 0; i < width; i++) {
        var rgb = shade(traceRay(b, panoramaDirection(i, j, width, height), step), true);
        var at = (j * width + i) * 4;
        out[at] = toByte(rgb[0]); out[at + 1] = toByte(rgb[1]); out[at + 2] = toByte(rgb[2]); out[at + 3] = 255;
      }
    }
    return out;
  }

  /* The same algorithm as a WebGPU compute shader: one invocation per pixel. */
  var WGSL = [
    "struct Params { v: f32, R: f32, sigma: f32, step: f32, width: u32, height: u32, maxSteps: u32, pad: u32 };",
    "@group(0) @binding(0) var<uniform> P: Params;",
    "@group(0) @binding(1) var<storage, read_write> pixels: array<u32>;",
    "const PI: f32 = 3.14159265358979;",
    "fn shape(r: f32) -> f32 { return (tanh(P.sigma * (r + P.R)) - tanh(P.sigma * (r - P.R))) / (2.0 * tanh(P.sigma * P.R)); }",
    "fn slope(r: f32) -> f32 { let a = tanh(P.sigma * (r + P.R)); let b = tanh(P.sigma * (r - P.R));",
    "  return P.sigma * ((1.0 - a * a) - (1.0 - b * b)) / (2.0 * tanh(P.sigma * P.R)); }",
    "fn rates(s: array<f32, 8>) -> array<f32, 8> {",
    "  let xs = s[1] - P.v * s[0]; let r = sqrt(xs * xs + s[2] * s[2] + s[3] * s[3]);",
    "  let f = shape(r); var overR = 0.0; if (r > 0.0) { overR = slope(r) / r; }",
    "  let W = s[4] + P.v * f * s[5]; let force = P.v * W * s[5] * overR;",
    "  return array<f32, 8>(-W, s[5] - P.v * f * W, s[6], s[7], force * (-P.v * xs), force * xs, force * s[2], force * s[3]); }",
    "fn axpy(s: array<f32, 8>, h: f32, k: array<f32, 8>) -> array<f32, 8> {",
    "  var o: array<f32, 8>; for (var i = 0; i < 8; i++) { o[i] = s[i] + h * k[i]; } return o; }",
    "fn skyColour(d: vec3<f32>) -> vec3<f32> {",
    "  let lon = atan2(d.y, d.x); let lat = asin(clamp(d.z, -1.0, 1.0)); let cell = PI / 12.0;",
    "  let checker = f32((i32(floor(lon / cell)) + i32(floor(lat / cell))) & 1);",
    "  let ahead = 0.5 * (1.0 + cos(lon) * cos(lat));",
    "  var rgb = (ahead * vec3<f32>(0.95, 0.55, 0.25) + (1.0 - ahead) * vec3<f32>(0.25, 0.55, 0.95)) * (0.55 + 0.45 * checker);",
    "  let st = PI / 6.0; var meridian = abs(lon / st - round(lon / st)) * cos(lat);",
    "  if (abs(lat) > 80.0 * PI / 180.0) { meridian = 1.0; }",
    "  if (min(meridian, abs(lat / st - round(lat / st))) < 0.02) { rgb = vec3<f32>(0.95); }",
    "  return rgb; }",
    "@compute @workgroup_size(8, 8)",
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {",
    "  if (id.x >= P.width || id.y >= P.height) { return; }",
    "  let lon = (f32(id.x) + 0.5) / f32(P.width) * 2.0 * PI - PI;",
    "  let lat = PI / 2.0 - (f32(id.y) + 0.5) / f32(P.height) * PI;",
    "  let d = vec3<f32>(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat));",
    "  var s = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 1.0 - P.v * shape(0.0) * d.x, d.x, d.y, d.z);",
    "  let edge = P.R + 20.0 / P.sigma; var colour = vec3<f32>(0.0); var done = false;",
    "  for (var n = 0u; n < P.maxSteps && !done; n++) {",
    "    let k1 = rates(s); let xs = s[1] - P.v * s[0]; let r = sqrt(xs * xs + s[2] * s[2] + s[3] * s[3]);",
    "    let rel = k1[1] - P.v * k1[0]; let speed = sqrt(rel * rel + k1[2] * k1[2] + k1[3] * k1[3]);",
    "    let h = min(P.step * exp(0.4 * P.sigma * abs(r - P.R)), 2.0) / max(speed, 1e-30);",
    "    let k2 = rates(axpy(s, 0.5 * h, k1)); let k3 = rates(axpy(s, 0.5 * h, k2)); let k4 = rates(axpy(s, h, k3));",
    "    for (var i = 0; i < 8; i++) { s[i] = s[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]); }",
    "    let xe = s[1] - P.v * s[0]; let re = sqrt(xe * xe + s[2] * s[2] + s[3] * s[3]);",
    "    if (abs(s[4]) > 1e6) { done = true; }",
    "    else if (re > edge) {",
    "      let p = vec3<f32>(s[5], s[6], s[7]); let D = 1.0 / s[4];",
    "      var rgb = skyColour(normalize(p)); let shift = tanh(log(D)); let a = abs(shift);",
    "      var tint = vec3<f32>(1.0, 0.35, 0.25); if (shift > 0.0) { tint = vec3<f32>(0.35, 0.55, 1.0); }",
    "      rgb = ((1.0 - 0.5 * a) * rgb + 0.5 * a * tint) * pow(clamp(D * D * D * D, 0.15, 3.0), 0.25);",
    "      colour = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)); done = true; }",
    "  }",
    "  let c = vec3<u32>(floor(colour * 255.0 + 0.5));",
    "  pixels[id.y * P.width + id.x] = c.x | (c.y << 8u) | (c.z << 16u) | (255u << 24u);",
    "}",
  ].join("\n");

  var api = {
    shape: shape,
    slope: slope,
    traceRay: traceRay,
    panoramaDirection: panoramaDirection,
    skyColour: skyColour,
    shade: shade,
    toByte: toByte,
    renderCPU: renderCPU,
    WGSL: WGSL,
    BLUESHIFT: BLUESHIFT,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.WarpRaytracer = api;
})(this);
