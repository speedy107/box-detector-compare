# box-detector-compare

A standalone blue/red box detector powered by the supplied **best.onnx** model. The model is bundled and loaded automatically when the site opens. No model upload is needed.

**Site:** https://speedy107.github.io/box-detector-compare/

Choose a photo or start your camera. Photos and camera frames are processed entirely in the browser. The page shows boxes, class confidence, counts, and inference time; Save detection downloads the annotated image. The confidence slider defaults to 35%.

## Model

`models/best.onnx` is the exact file supplied by the user. Input is float32 RGB `[1,3,320,320]`, letterboxed with 114 padding and divided by 255. Output is `[1,6,2100]`: box center x/y, width/height, blue score (class 0), red score (class 1). The decoder uses class-aware NMS at IoU 0.45. Only this model is loaded; there is no V5 comparison or model picker.

## Run locally

```sh
python -m http.server 8001 --bind 127.0.0.1
```

No build step or package installation is required. ONNX Runtime Web 1.20.1 WASM is included. Camera access requires HTTPS or a browser-trusted local origin and user permission. Stop camera pauses processing. Runtime sessions warm up before measured inference; timings are approximate device measurements.

## GitHub Pages

Settings → Pages → Deploy from a branch → main → / (root). Pushes to main update the site automatically.

## Browser validation

Chromium checks cover automatic loading of the exact bundled model, known blue/red examples, a negative background, invalid photo recovery, simulated camera start/stop, mobile layout, confidence adjustment, and detection download. Physical phone-camera performance is not measured.

With Python Playwright and Chromium installed, start the local server, then run:

```sh
python tests/browser_smoke.py --blue /path/to/one-blue-box.jpg \
  --red /path/to/one-red-box.jpg --negative /path/to/no-boxes.jpg
```

Use `--chromium /usr/bin/chromium` to run with an existing Chromium executable. Test photos should contain the expected objects; they are not included in this repository.

## Licensing

The model uses Ultralytics YOLO under AGPL-3.0/commercial terms; see LICENSE-YOLO.txt. This site's code follows AGPL-3.0 in that file. ONNX Runtime's license is in vendor/ONNX-Runtime-LICENSE.txt.
