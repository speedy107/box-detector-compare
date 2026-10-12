# box-detector-compare


A static website for comparing an uploaded ONNX blue/red box detector with [V5](https://github.com/speedy107/box-detector-v5). Photos, camera frames, and uploaded models remain in the browser.

Choose an ONNX file, then a photo or Start camera. Both models analyze the exact same frozen frame with 320×320 letterbox preprocessing, RGB/255 input, shared confidence, and class-aware NMS at IoU 0.45. Results show detections, confidence, inference time, and matching predictions at IoU ≥0.5. Agreement measures similarity between detectors, not accuracy.

Compatible uploads have one float32 input `[1,3,320,320]` and one YOLO detection output `[1,6,N]`: center x/y, width/height, blue score, red score. The uploaded `best.onnx` supplied during development was validated with these shapes and class order. It is not bundled or published; select it locally using Choose ONNX. Models with other input sizes, class orders, objectness channels, or end-to-end NMS outputs are unsupported.

## Run locally

```sh
python -m http.server 8000 --bind 127.0.0.1
```

No build step or package installation is needed. ONNX Runtime Web 1.20.1 WASM and the V5 model are included. Camera access requires HTTPS or a browser-trusted local origin and user permission. Stop camera pauses processing; Save comparison downloads both annotated panels as a PNG. Models are initialized with a warmup run before timing; device timings are approximate, not a standardized benchmark.

## GitHub Pages

Enable Settings → Pages → Deploy from a branch → main → / (root). The site is served directly; no custom workflow or build is needed.

## Browser validation

The site was tested in Chromium with the supplied `best.onnx`, known blue/red examples, a negative background, an invalid upload, a simulated camera, mobile layout, and comparison download. This is not a physical phone-camera performance measurement.

For repeatable browser checks, install Python Playwright and its Chromium browser, start the local server on port 8001, then run:

```sh
python tests/browser_smoke.py --model /path/to/best.onnx \
  --blue /path/to/one-blue-box.jpg --red /path/to/one-red-box.jpg
```

Use `--chromium /usr/bin/chromium` to run with an existing Chromium executable. The photos should each contain exactly one clearly visible box. Test photos and uploaded models are not included in this repository.

## Licensing

The V5 model uses Ultralytics YOLO under AGPL-3.0/commercial terms; see LICENSE-YOLO.txt. ONNX Runtime's license is in vendor/ONNX-Runtime-LICENSE.txt. This site's code follows AGPL-3.0 in LICENSE-YOLO.txt. Source-image rights and evaluation limitations are described in the original V5 repository.
