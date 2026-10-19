"""End-to-end tests using real YOLO/OpenCV and a controlled camera stream.

The scene geometry is synthetic. Timings describe this Chromium container,
not the iQOO Neo 6, and do not establish real-world distance accuracy.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
SCENES = json.loads((ROOT / "tests/fixtures/motion/scenes.json").read_text())


def camera_script(fixture_prefix="../tests/fixtures/motion/"):
    """Deliver genuinely fresh video frames, including when a scene is static."""
    return """(() => {
      const prefix = PREFIX;
      navigator.mediaDevices.getUserMedia = async () => {
        const canvas = document.createElement('canvas');
        let image, portrait = false, stream;
        canvas.width = 640; canvas.height = 480;
        window.__qaZoom = 1;
        const draw = () => {
          const ctx = canvas.getContext('2d');
          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          if (portrait) { ctx.translate(480, 0); ctx.rotate(Math.PI / 2); }
          if (image) ctx.drawImage(image, 0, 0);
        };
        window.setScene = async (name, nextPortrait = false) => {
          const nextImage = await createImageBitmap(await (await fetch(new URL(prefix + name, location.href))).blob());
          if (image) image.close();
          image = nextImage; portrait = nextPortrait;
          canvas.width = portrait ? 480 : 640;
          canvas.height = portrait ? 640 : 480;
          draw();
          const video = document.querySelector('#video');
          if (video.srcObject && video.requestVideoFrameCallback) {
            await new Promise(resolve => video.requestVideoFrameCallback(resolve));
          }
        };
        await window.setScene('motion-0.jpg');
        stream = canvas.captureStream(30);
        const track = stream.getVideoTracks()[0];
        track.getSettings = () => ({deviceId:'controlled-cube-camera',
          width:canvas.width, height:canvas.height, zoom:window.__qaZoom,
          facingMode:'environment'});
        const tick = () => {
          if (track.readyState === 'ended') { image.close(); return; }
          draw(); requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
        return stream;
      };
    })();""".replace("PREFIX", json.dumps(fixture_prefix))


def controllable_replies_script():
    """Delay delivery of genuine inference replies, without replacing inference."""
    return """(() => {
      const NativeWorker = window.Worker;
      window.__qaHoldReplies = false;
      window.__qaHeldReplies = [];
      window.releaseReplies = () => {
        window.__qaHoldReplies = false;
        const pending = window.__qaHeldReplies.splice(0);
        for (const deliver of pending) deliver();
      };
      window.Worker = class ControlledWorker extends NativeWorker {
        constructor(...args) {
          super(...args);
          let messageHandler = null;
          Object.defineProperty(this, 'onmessage', {
            configurable:true,
            get:() => messageHandler,
            set:handler => { messageHandler = handler; }
          });
          this.addEventListener('message', event => {
            const deliver = () => messageHandler?.call(this, event);
            if (window.__qaHoldReplies && Array.isArray(event.data?.result?.boxes)) {
              window.__qaHeldReplies.push(deliver);
            } else deliver();
          });
        }
      };
    })();"""


def sample_frames(page, selector="#latency", count=20, cadence_selector="#cadence"):
    """Sample completed results and main-thread animation availability together."""
    return page.evaluate("""({selector, count, cadenceSelector}) => new Promise((resolve, reject) => {
      const values = [], animationGaps = [], element = document.querySelector(selector);
      let previous = element.dataset.frameNumber, previousResult = performance.now();
      let previousAnimation, active = true;
      const animate = now => {
        if (!active) return;
        if (previousAnimation !== undefined) animationGaps.push(now - previousAnimation);
        previousAnimation = now; requestAnimationFrame(animate);
      };
      requestAnimationFrame(animate);
      const timeout = setTimeout(() => {
        active = false; observer.disconnect(); reject(new Error('Completed-frame sample timeout'));
      }, 60000);
      const observer = new MutationObserver(() => {
        const frame = element.dataset.frameNumber;
        if (!frame || frame === previous) return;
        previous = frame;
        const now = performance.now();
        const cadence = document.querySelector(cadenceSelector);
        values.push({frame:Number(frame), update_ms:Number(element.dataset.totalMs),
          pose_ms:Number(element.dataset.distanceMs || 0),
          result_interval_ms:now - previousResult,
          app_reported_interval_ms:cadence && cadence.dataset.intervalMs ? Number(cadence.dataset.intervalMs) : null});
        previousResult = now;
        if (values.length === count) {
          clearTimeout(timeout); observer.disconnect(); active = false;
          resolve({frames:values, animation_gaps_ms:animationGaps});
        }
      });
      observer.observe(element, {attributes:true});
    })""", {"selector": selector, "count": count, "cadenceSelector": cadence_selector})


def stats(values):
    if not values:
        return {"samples": 0}
    ordered = sorted(values)
    return {"samples": len(values), "median": round(statistics.median(ordered), 2),
            "p95": round(ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], 2),
            "maximum": round(max(ordered), 2)}


def timing_summary(samples):
    frames = samples["frames"]
    return {"capture_to_result_ms": stats([f["update_ms"] for f in frames]),
            "result_interval_ms": stats([f["result_interval_ms"] for f in frames][1:]),
            "pose_ms": stats([f["pose_ms"] for f in frames]),
            "main_thread_animation_gap_ms": stats(samples["animation_gaps_ms"]),
            "raw": samples}


def frame_number(page):
    return int(page.locator("#latency").get_attribute("data-frame-number") or 0)


def wait_frames(page, previous, count=3):
    page.wait_for_function("([previous,count]) => Number(document.querySelector('#latency').dataset.frameNumber) >= previous + count",
                           arg=[previous, count], timeout=30000)


def read_distance(page, scene):
    text = page.locator("#detections").inner_text()
    found = re.search(r"≈\s*([0-9.]+)\s*m", text)
    assert found, f"{scene}: {text}"
    assert "reference angle" not in text.lower()
    return float(found.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8003/pro/")
    parser.add_argument("--chromium", default="/usr/bin/chromium")
    parser.add_argument("--report", type=Path, default=ROOT / "pro/validation/browser.json")
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    observations = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium, headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 390, "height": 844}, permissions=["camera"])
        page = context.new_page()
        errors, requests = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url))
        page.add_init_script(camera_script())
        page.add_init_script(controllable_replies_script())
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function("document.querySelector('#modelState').textContent.includes('Ready')", timeout=60000)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Mobile viewport overflows horizontally"
        page.locator("#start").click()
        page.wait_for_function("document.querySelector('#video').srcObject !== null", timeout=30000)
        page.wait_for_function("Number(document.querySelector('#latency').dataset.frameNumber) > 0", timeout=60000)
        assert "≈" not in page.locator("#detections").inner_text(), "Uncalibrated distance must not be invented"
        page.locator("#setupPanel").evaluate("element => element.open = true")
        page.locator("#referenceMetres").fill("3.00")
        page.evaluate("() => { document.querySelector('#calibrate').click(); document.querySelector('#calibrate').click(); }")
        wait_frames(page, frame_number(page))
        assert page.evaluate("localStorage.getItem('cube-lens-pro-reference-v1') === null"), "Cancelled setup persisted a partial reference"
        assert "≈" not in page.locator("#detections").inner_text()
        page.locator("#calibrate").click()
        page.wait_for_function("document.querySelector('#calibrationState').textContent.includes('Camera reference saved')", timeout=60000)
        saved = page.evaluate("JSON.parse(localStorage.getItem('cube-lens-pro-reference-v1'))")
        assert saved and saved["sideMetres"] == .35, saved
        page.wait_for_function("document.querySelector('#detections').textContent.includes('≈')", timeout=30000)
        page.locator("#setupPanel").evaluate("element => element.open = false")

        for scene in SCENES:
            previous = frame_number(page)
            page.evaluate("name => window.setScene(name)", scene["name"])
            wait_frames(page, previous)
            metres = read_distance(page, scene["name"])
            error = abs(metres / scene["truth_metres"] - 1)
            assert error < .1, (scene["name"], scene["angles"], scene["truth_metres"], metres)
            observations.append({"scene": scene["name"], "camera_angles_degrees": scene["angles"],
                                 "synthetic_surface_metres": scene["truth_metres"], "estimated_metres": metres,
                                 "relative_error": error})

        previous = frame_number(page)
        page.evaluate("() => window.setScene('motion-1.jpg', true)")
        wait_frames(page, previous)
        portrait_distance = read_distance(page, "portrait rotation")
        assert abs(portrait_distance / 3 - 1) < .1, portrait_distance
        observations.append({"scene": "motion-1.jpg rotated 90 degrees to portrait", "synthetic_surface_metres": 3,
                             "estimated_metres": portrait_distance, "relative_error": abs(portrait_distance / 3 - 1)})

        previous = frame_number(page)
        page.evaluate("() => window.setScene('../negative.jpg')")
        wait_frames(page, previous)
        assert "≈" not in page.locator("#detections").inner_text(), "A cube-free scene retained a previous distance"
        previous = frame_number(page)
        page.evaluate("() => window.setScene('motion-0.jpg')")
        wait_frames(page, previous)
        assert abs(read_distance(page, "benchmark scene") / 3 - 1) < .1

        runtime_state = page.evaluate("""() => ({
          cross_origin_isolated:window.crossOriginIsolated,
          threads:Number(document.querySelector('#modelState').dataset.threads),
          backend:document.querySelector('#modelState').dataset.backend || 'wasm',
          service_worker_script:navigator.serviceWorker.controller?.scriptURL || null
        })""")
        assert runtime_state["cross_origin_isolated"], "Scoped isolation service worker did not activate"
        assert runtime_state["threads"] >= 1, runtime_state
        samples = sample_frames(page, count=args.frames)
        assert all(f["update_ms"] < 1000 for f in samples["frames"]), samples
        assert all(f["result_interval_ms"] < 1000 for f in samples["frames"]), samples
        assert page.evaluate("Number.parseInt(document.querySelector('#freshness').textContent) >= Number.parseInt(document.querySelector('#latency').textContent)"), "Reading age understates the current frame's processing time"
        assert not any("/models/depth-" in url for url in requests), "Slow neural depth model should not be requested"
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(args.report.with_suffix(".png")), full_page=True)

        page.evaluate("() => { window.__qaHoldReplies = true; }")
        page.wait_for_function("window.__qaHeldReplies.length > 0", timeout=10000)
        page.wait_for_function("document.querySelector('#qualityState').textContent.includes('Waiting for a fresh frame')", timeout=5000)
        assert "≈" not in page.locator("#detections").inner_text(), "Stale worker replies left a numeric distance visible"
        assert page.evaluate("(() => { const c=document.querySelector('#overlay'); const a=c.getContext('2d').getImageData(0,0,c.width,c.height).data; return a.every(v => v === 0); })()"), "Stale overlay was not cleared"
        previous = frame_number(page)
        page.evaluate("() => window.releaseReplies()")
        wait_frames(page, previous, 2)
        read_distance(page, "recovered from held reply")

        page.evaluate("() => { window.__qaHoldReplies = true; }")
        page.wait_for_function("window.__qaHeldReplies.length > 0", timeout=10000)
        previous = frame_number(page)
        page.evaluate("() => { document.querySelector('#stop').click(); document.querySelector('#start').click(); }")
        page.wait_for_function("document.querySelector('#video').srcObject !== null", timeout=10000)
        assert "≈" not in page.locator("#detections").inner_text(), "Restart retained the previous camera measurement"
        page.evaluate("() => window.releaseReplies()")
        wait_frames(page, previous, 2)
        read_distance(page, "rapid restart while inference was pending")

        page.locator("#stop").click()
        page.wait_for_function("document.querySelector('#video').srcObject === null && !document.querySelector('#start').disabled")
        previous = frame_number(page)
        page.wait_for_timeout(250)
        assert frame_number(page) == previous, "Processing continued after Stop"
        page.reload(wait_until="networkidle")
        page.wait_for_function("document.querySelector('#modelState').textContent.includes('Ready')", timeout=60000)
        assert page.evaluate("localStorage.getItem('cube-lens-pro-reference-v1') !== null"), "Saved reference was lost on reload"
        page.locator("#start").click()
        page.wait_for_function("document.querySelector('#video').srcObject !== null")
        page.wait_for_function("document.querySelector('#detections').textContent.includes('≈')", timeout=30000)
        previous = frame_number(page)
        page.evaluate("() => { window.__qaZoom = 2; }")
        wait_frames(page, previous)
        assert "≈" not in page.locator("#detections").inner_text(), "Zoom change must invalidate the camera reference"
        assert re.search(r"recalibr|reference|zoom", page.locator("#qualityState").inner_text(), re.I), page.locator("#qualityState").inner_text()
        page.locator("#setupPanel").evaluate("element => element.open = true")
        page.locator("#clearCalibration").click()
        assert page.evaluate("localStorage.getItem('cube-lens-pro-reference-v1') === null")
        assert "≈" not in page.locator("#detections").inner_text()
        page.locator("#stop").click()
        page.wait_for_function("document.querySelector('#video').srcObject === null")
        page.wait_for_function("!document.querySelector('#photoFile').disabled", timeout=30000)

        page.locator("#photoFile").set_input_files({"name":"invalid.jpg", "mimeType":"image/jpeg", "buffer":b"not an image"})
        page.wait_for_function("document.querySelector('#status').textContent.includes('Could not open this photo')")
        assert not page.locator("#start").is_disabled(), "Malformed photo left camera controls disabled"
        page.locator("#photoFile").set_input_files(ROOT / "tests/fixtures/blue.jpg")
        page.wait_for_function("document.querySelector('#detections').textContent.toLowerCase().includes('blue')", timeout=30000)
        assert "≈" not in page.locator("#detections").inner_text(), "A photo with unknown camera intrinsics should not inherit a camera reference"
        page.wait_for_function("!document.querySelector('#save').disabled")
        with page.expect_download() as download_info:
            page.locator("#save").click()
        download = download_info.value
        assert download.failure() is None
        assert download.suggested_filename.lower().endswith(".png")
        page.locator("#about").evaluate("element => element.open = true")
        with page.expect_download() as diagnostics_info:
            page.locator("#diagnostics").click()
        diagnostics_download = diagnostics_info.value
        assert diagnostics_download.failure() is None
        diagnostics = json.loads(Path(diagnostics_download.path()).read_text())
        assert diagnostics["version"] == "cube-lens-pro-1"
        assert diagnostics["frames"], "Diagnostics should retain recent timing and quality samples"
        assert "data:image" not in json.dumps(diagnostics), "Diagnostics unexpectedly contain camera image data"
        failure_context = browser.new_context(viewport={"width":390,"height":844}, service_workers="block")
        failure_page = failure_context.new_page()
        failure_page.add_init_script("navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Denied by test', 'NotAllowedError'); };")
        failure_page.goto(args.url, wait_until="networkidle")
        failure_page.wait_for_function("document.querySelector('#modelState').textContent.includes('Ready')", timeout=60000)
        assert failure_page.evaluate("!window.crossOriginIsolated")
        assert failure_page.evaluate("Number(document.querySelector('#modelState').dataset.threads) === 1"), "Blocked service workers did not select single-thread fallback"
        failure_page.locator("#photoFile").set_input_files(ROOT / "tests/fixtures/blue.jpg")
        failure_page.wait_for_function("document.querySelector('#detections').textContent.toLowerCase().includes('blue')", timeout=30000)
        failure_page.locator("#start").click()
        failure_page.wait_for_function("document.querySelector('#status').textContent.includes('Camera permission denied')")
        assert not failure_page.locator("#start").is_disabled()
        assert failure_page.evaluate("document.querySelector('#video').srcObject === null")
        failure_context.close()
        page.goto(urljoin(args.url, "../"), wait_until="domcontentloaded")
        assert page.evaluate("!window.crossOriginIsolated"), "Pro isolation escaped its dedicated site path"
        assert page.evaluate("navigator.serviceWorker.controller === null"), "Pro service worker controls the original site"
        assert "Find the cubes" in page.locator("h1").inner_text()
        assert not errors, errors
        browser.close()

    report = {"passed": True, "date": datetime.now(timezone.utc).isoformat(),
              "runtime": "Headless Chromium container; real browser YOLO and OpenCV workers; synthetic 30 fps camera",
              "runtime_configuration": runtime_state,
              "timing": timing_summary(samples), "viewpoints": observations,
              "checks": ["35 cm cube reference", "all ten yaw/pitch/roll/distance scenes", "portrait rotation preserves reference",
                         "distance hidden before calibration", "cancelled reference never persists", "zoom change invalidates reference", "stop and restart",
                         "rapid restart during pending inference", "stale overlay and numeric cards expire", "delayed worker reply recovers",
                         "reference survives reload", "clear reference", "malformed photo recovers", "camera permission denial recovers",
                         "photo detection without assumed intrinsics", "PNG export", "diagnostics export",
                         "scoped service worker isolation", "blocked service worker single-thread photo fallback", "original site remains uncontrolled and unisolated",
                         "mobile layout", "no stale distance on cube-free scene", "no neural depth download", "no page errors"],
              "limitations": "Synthetic, clear-edged cubes and a controlled camera stream. Not measured on iQOO Neo 6. These results do not establish physical accuracy or phone latency."}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "timing"}, indent=2))
    print(json.dumps({key: value for key, value in report["timing"].items() if key != "raw"}, indent=2))


if __name__ == "__main__":
    main()
