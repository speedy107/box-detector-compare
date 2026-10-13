# box-detector-compare

A standalone blue/red box detector powered by the supplied **best.onnx** model. The model is bundled and loaded automatically when the site opens. No model upload is needed.

**Site:** https://speedy107.github.io/box-detector-compare/

Choose a photo or start your camera. Photos and camera frames are processed entirely in the browser. The page shows boxes, class confidence, counts, and inference time; Save detection downloads the annotated image. The confidence slider defaults to 35%.

## Indoor distance estimates

The depth integration uses **Depth Anything V2 Metric Hypersim Small**, which predicts optical-axis depth to the visible surface in metres. This is the indoor metric checkpoint, not the relative-depth model. It is a learned estimate; validate it against measured distances using the camera and scenes where you will use it. It does not compute exact Euclidean range without camera intrinsics.

When Estimate distance is enabled, depth and YOLO process the same frozen frame. The per-block estimate is the median valid depth in the central 60% of its detected box, to reduce background contamination. Sparse, invalid, or saturated estimates display Depth unavailable. The indoor training cap is 20 metres. Depth runs in a separate browser worker and makes camera updates slower. Turn it off for detection-only use.

The browser model uses whole-frame square resizing to 518×518 and ImageNet RGB normalization, independently of YOLO's letterbox. The ONNX export uses per-channel int8 constant MatMul weights, retaining other operations in float32. Runtime model integrity is checked against the export metadata.

The verified depth artifact is bundled at `models/depth-anything-v2-metric-indoor-q8.onnx` (35,534,766 bytes), with its `.json` provenance and numeric validation report. The official checkpoint checksum and pinned source were verified before export. Across four validation images, FP32 export error stayed below 0.0001 metres and mean Q8 relative error ranged from 0.64% to 1.81%. These compare model implementations; they do not measure physical distance accuracy. If depth is unavailable, block detection remains usable and the page offers Retry depth. A three-minute timeout releases a stalled worker, and Stop cancels active camera depth work.

At least half of the pixels sampled inside a block must have valid depth. Too few valid samples or estimates near the 20-metre cap display Depth unavailable.

## Model

`models/best.onnx` is the exact file supplied by the user. Input is float32 RGB `[1,3,320,320]`, letterboxed with 114 padding and divided by 255. Output is `[1,6,2100]`: box center x/y, width/height, blue score (class 0), red score (class 1). The decoder uses class-aware NMS at IoU 0.45. Only this model is loaded; there is no V5 comparison or model picker.

## Run locally

```sh
python -m http.server 8001 --bind 127.0.0.1
```

No build step or package installation is required. ONNX Runtime Web 1.20.1 WASM is included. Camera access requires HTTPS or a browser-trusted local origin and user permission. Stop camera pauses processing. YOLO warms up before measured inference. The first depth timing can include runtime initialization; timings are approximate device measurements.

## GitHub Pages

Settings → Pages → Deploy from a branch → main → / (root). Pushes to main update the site automatically.

## Browser validation

Chromium checks cover automatic loading of the exact bundled model, known blue/red examples, a negative background, invalid photo recovery, simulated camera start/stop, mobile layout, confidence adjustment, and detection download. Physical phone-camera performance is not measured.

With Python Playwright and Chromium installed, start the local server, then run:

```sh
python tests/browser_smoke.py --blue /path/to/one-blue-box.jpg \
  --red /path/to/one-red-box.jpg --negative /path/to/no-boxes.jpg
```

Use `--chromium /usr/bin/chromium` to run with an existing Chromium executable. The included `tests/fixtures/blue.jpg`, `red.jpg`, and `negative.jpg` make the checks reproducible. The block photos are composites of existing real block cutouts over a generated indoor background; the fixtures do not establish physical distance accuracy.

```sh
node tests/depth.test.mjs
node tests/depth-worker.test.mjs
python tests/browser_smoke.py --chromium /usr/bin/chromium \
  --blue tests/fixtures/blue.jpg --red tests/fixtures/red.jpg \
  --negative tests/fixtures/negative.jpg
python tests/depth_browser_smoke.py --chromium /usr/bin/chromium \
  --blue tests/fixtures/blue.jpg --red tests/fixtures/red.jpg \
  --report validation/depth-browser.json
```

Depth checks run the real bundled model, exercise metres and the distance toggle, reject mismatched checksums, recover through Retry depth, and run depth on a simulated camera showing a real blue block. They also hold a model download open, stop the camera, and verify detection can resume. `validation/` contains the completed checks. Phone performance and physical camera accuracy remain unmeasured.

## Rebuild the depth artifact

This is optional; the application already contains the browser model. Use Python 3.12 and `scripts/requirements-export.txt`. Install the CPU PyTorch wheels first:

```sh
python -m venv .venv
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r scripts/requirements-export.txt
git clone https://github.com/DepthAnything/Depth-Anything-V2.git /tmp/Depth-Anything-V2
git -C /tmp/Depth-Anything-V2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
curl -fL -o /tmp/depth_anything_v2_metric_hypersim_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/3bc65d4e14a6786a61acec16453c50e12bf5f338/depth_anything_v2_metric_hypersim_vits.pth
.venv/bin/python scripts/export_metric_depth.py \
  --source /tmp/Depth-Anything-V2 \
  --checkpoint /tmp/depth_anything_v2_metric_hypersim_vits.pth \
  --sha256 b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545 \
  --checkpoint-revision 3bc65d4e14a6786a61acec16453c50e12bf5f338 \
  --validation-image tests/fixtures/blue.jpg \
  --validation-image tests/fixtures/red.jpg \
  --validation-image tests/fixtures/negative.jpg \
  --validation-image tests/fixtures/workshop_clutter.png
```

Keep the FP32 intermediate out of Git and distribution packages. Rerun numeric validation and browser checks whenever the depth artifact changes.

## Licensing

The model uses Ultralytics YOLO under AGPL-3.0/commercial terms; see LICENSE-YOLO.txt. This site's code follows AGPL-3.0 in that file. ONNX Runtime's license is in vendor/ONNX-Runtime-LICENSE.txt.

Depth Anything V2 Small is Apache-2.0; see vendor/Depth-Anything-V2-LICENSE.txt. The upstream project is https://github.com/DepthAnything/Depth-Anything-V2 and the indoor checkpoint is https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small.
