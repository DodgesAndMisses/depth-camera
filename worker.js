// worker.js — all ML lives here, off the main thread.
// Same-origin module worker: iOS Safari refuses ORT's built-in cross-origin
// proxy worker, and killed the page (watchdog) when ORT session init blocked
// the main thread. In a worker there is no watchdog and no UI to freeze.

import { pipeline, RawImage, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.backends.onnx.wasm.proxy = false;   // we ARE the worker
env.backends.onnx.wasm.numThreads = 1;  // iOS Safari: pthread workers fail to load

// --- Colormap: far (deep blue) -> mid (teal) -> near (yellow), 256-entry LUT
const LUT = new Uint8ClampedArray(256 * 3);
const stops = [
  [0.0,   0,  26, 128],   // far
  [0.55,  0, 179, 164],   // mid
  [1.0, 255, 204,   0],   // near
];
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  let a = stops[0], b = stops[stops.length - 1];
  for (let s = 0; s < stops.length - 1; s++)
    if (t >= stops[s][0] && t <= stops[s + 1][0]) { a = stops[s]; b = stops[s + 1]; break; }
  const f = (t - a[0]) / (b[0] - a[0] || 1);
  LUT[i * 3]     = a[1] + (b[1] - a[1]) * f;
  LUT[i * 3 + 1] = a[2] + (b[2] - a[2]) * f;
  LUT[i * 3 + 2] = a[3] + (b[3] - a[3]) * f;
}

let estimator = null;

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === "load") {
    const opts = {
      progress_callback: (p) => {
        if (p.status === "progress" && p.total)
          postMessage({ type: "progress", pct: Math.round((p.loaded / p.total) * 100) });
      },
    };
    try {
      let backend = "wasm";
      if (msg.tryWebGPU && typeof navigator !== "undefined" && navigator.gpu) {
        try {
          estimator = await pipeline("depth-estimation",
            "onnx-community/depth-anything-v2-small",
            { ...opts, device: "webgpu", dtype: "fp16" });
          backend = "webgpu";
        } catch { /* fall through to wasm */ }
      }
      if (!estimator) {
        estimator = await pipeline("depth-estimation",
          "onnx-community/depth-anything-v2-small",
          { ...opts, device: "wasm", dtype: msg.dtype || "q8",
            // Trade speed for peak RAM — iOS kills the tab at the
            // memory high-water mark at the end of session init.
            session_options: {
              enableCpuMemArena: false,
              enableMemPattern: false,
            } });
      }
      // Shrink the model's internal working resolution. The image processor
      // resizes ALL inputs to 518x518 by default, so downscaling the camera
      // frame alone saves nothing. 252 (multiple of 14) is ~4x less compute.
      if (msg.inferSize) {
        try {
          const ip = estimator.processor?.image_processor ?? estimator.processor;
          if (ip) ip.size = { width: msg.inferSize, height: msg.inferSize };
        } catch { /* non-fatal: falls back to default resolution */ }
      }
      postMessage({ type: "ready", backend });
    } catch (err) {
      postMessage({ type: "error",
        message: (err && (err.message || String(err))) || "unknown error" });
    }
    return;
  }

  if (msg.type === "frame") {
    if (!estimator) { postMessage({ type: "frameError", message: "model not ready" }); return; }
    try {
      const t0 = performance.now();
      const image = new RawImage(
        new Uint8ClampedArray(msg.buffer), msg.width, msg.height, 4);
      const { depth } = await estimator(image);   // grayscale RawImage, bright = near

      const d = depth.data, n = d.length;
      const out = new Uint8ClampedArray(n * 4);
      for (let i = 0; i < n; i++) {
        const v = d[i] * 3;
        out[i * 4]     = LUT[v];
        out[i * 4 + 1] = LUT[v + 1];
        out[i * 4 + 2] = LUT[v + 2];
        out[i * 4 + 3] = 255;
      }
      postMessage(
        { type: "depth", buffer: out.buffer, width: depth.width, height: depth.height,
          ms: Math.round(performance.now() - t0) },
        [out.buffer]);
    } catch (err) {
      postMessage({ type: "frameError",
        message: (err && (err.message || String(err))) || "unknown error" });
    }
  }
};

postMessage({ type: "boot" });
