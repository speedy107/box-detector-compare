# Cube Lens Pro

A separate camera application at `https://speedy107.github.io/box-detector-compare/pro/`.
The original application at the repository root is unchanged.

Use stationary blue/red **35 × 35 × 35 cm cubes**. Start the rear camera in Chrome, show one complete cube, and set a measured lens-to-nearest-surface reference. Seven clear frames establish the camera scale. After setup, the phone can move and change viewing angle, including portrait/landscape rotation. Keep the same lens, zoom and stream format; changes invalidate the reference.

## Pipeline

- One dedicated worker runs the supplied YOLO model and OpenCV cube pose on the same captured image. The camera preview stays live independently.
- Only one inference runs at a time. When it finishes, the next available video frame is captured. There is no queue or fixed 100 ms processing delay.
- RGB boundary gradients and robust fitted lines refine the colour silhouette to subpixel corners. Several bounded silhouette candidates compete using boundary support and reprojection error.
- Clipped, concave, low-confidence and unsupported outlines return an explanation instead of a distance. Readings and overlays expire after 750 ms without a fresh result.
- Distance is fitted independently in each frame; there is no fixed reference-angle constraint or temporal averaging that hides movement.
- A scoped `/pro/` service worker adds isolation headers for optional two-thread WASM. Unsupported or failed isolation falls back to one thread. It stores no camera images or asset cache and does not control the original root application. First launch may reload once before model/camera startup.
- Photos support detection; they do not inherit live-camera intrinsics. Save uses the exact analysed image. Diagnostics export recent timings/fit metrics, without images or the camera device ID.

The detector and runtime assets are reused from `../models/` and `../vendor/`; no neural depth model is downloaded. The detector, OpenCV and ONNX Runtime licences remain in the parent directories.

## Verification

Serve the repository root, then run:

```sh
python -m http.server 8003
node pro/tests/inference.test.mjs
node pro/tests/vision.test.mjs
node tests/pose.test.mjs
python pro/tests/browser_smoke.py --url http://127.0.0.1:8003/pro/
```

The browser check uses real model inference and real OpenCV on independently rendered cubes through a synthetic camera. It covers changing yaw/pitch/roll/range, portrait rotation, calibration persistence/cancellation, zoom mismatch, worker freshness, rapid stop/restart, failure recovery and exports. The pixel tests additionally cover movement blur, pale colours, clipping and occlusion.

Reports live in `validation/`. The original-site baseline and Pro timings use the same 30 fps synthetic scene. The thread benchmark uses actual browser isolation and confirms pthread workers; Chrome page CPU throttling is not used to imply worker hardware performance.

**These are container benchmarks and synthetic accuracy checks, not measurements on an iQOO Neo 6.** Actual accuracy depends on accurate cube dimensions/reference measurement, edge visibility, lens distortion, blur, lighting and camera processing. Image-fit quality is not a physical accuracy probability. Validate at several measured distances and viewpoints. Processing time means application capture-to-display work, not network ping or total sensor latency.

Final same-scene browser comparison: median capture-to-display **57.15 ms** versus **70.95 ms**; median update interval **57.34 ms** versus **170.9 ms** (about **3×** the update frequency). Main-thread animation gaps stayed at about **16.7 ms**, versus a previous 50 ms 95th percentile. These are the container results recorded in `validation/browser.json` and `validation/baseline.json`.
