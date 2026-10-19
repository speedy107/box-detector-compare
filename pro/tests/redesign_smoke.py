"""UI/workflow checks for the instrument-workstation redesign.

Inference, calibration and downloads are exercised in the browser. The controlled
camera contains synthetic cube scenes; this does not measure physical accuracy.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from playwright.sync_api import sync_playwright

from browser_smoke import ROOT, camera_script, frame_number, read_distance, wait_frames


def assert_no_overflow(page):
    dimensions = page.evaluate("""() => ({width:innerWidth,
      document:document.documentElement.scrollWidth, body:document.body.scrollWidth})""")
    assert max(dimensions["document"], dimensions["body"]) <= dimensions["width"], dimensions
    return dimensions


def save_screenshot(page, path):
    # Full-page captures should begin at the origin so fixed, translated-offscreen
    # accessibility controls are captured in their actual resting position.
    page.evaluate("window.scrollTo(0, 0)")
    page.screenshot(path=str(path), full_page=True)


def geometry(page):
    return page.evaluate("""() => {
      const rect = selector => {
        const element = document.querySelector(selector);
        const {x,y,width,height,bottom,right} = element.getBoundingClientRect();
        return {x,y,width,height,bottom,right};
      };
      return {viewer:rect('#viewer'), inspector:rect('aside'), reference:rect('#setupPanel'),
        camera:rect('#viewer').width / rect('aside').width};
    }""")


def open_details_for(page, selector):
    control = page.locator(selector)
    ancestor = control.locator("xpath=ancestor::details[1]")
    if ancestor.count() and not ancestor.evaluate("element => element.open"):
        ancestor.locator(":scope > summary").focus()
        page.keyboard.press("Enter")
    assert control.is_visible(), f"{selector} is not accessible after opening its controls"
    return ancestor


def assert_detection_colour(page, name):
    selector = f"#detections .{name}"
    page.wait_for_selector(selector)
    label = page.locator(selector).locator(".detection-label")
    colour = label.evaluate("element => getComputedStyle(element).color")
    values = [float(value) for value in re.findall(r"[0-9.]+", colour)[:3]]
    assert len(values) == 3, colour
    red, _, blue = values
    assert blue > red + 20 if name == "blue" else red > blue + 20, (name, colour)
    return colour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8003/pro/")
    parser.add_argument("--chromium", default="/usr/bin/chromium")
    parser.add_argument("--report", type=Path, default=ROOT / "pro/validation/redesign/report.json")
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    screenshots = []
    layouts = {}
    events = []
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chromium, headless=True,
                                             args=["--no-sandbox"])
        context = browser.new_context(viewport={"width":1440,"height":1000}, permissions=["camera"])
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script(camera_script())
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function("document.querySelector('#modelState').textContent.includes('Ready')", timeout=60000)
        assert page.locator("#stop").is_disabled(), "Stop should be disabled before the camera starts"
        assert page.locator("#save").is_disabled(), "No frame exists to export before camera/photo input"
        assert "≈" not in page.locator("#detections").inner_text(), "Idle UI invented a distance"
        assert page.locator("#latency").inner_text().strip() == "—", "Idle processing readout is fabricated"
        assert page.locator("#diagnostics").is_visible(), "Diagnostics is unavailable from the command area"
        assert not page.locator("#diagnostics").locator("xpath=ancestor::details").count(), "Diagnostics is buried in help"
        assert_no_overflow(page)

        # Use keyboard focus, then inspect its actual rendered treatment.
        page.locator("#start").focus()
        page.keyboard.press("Tab")
        page.keyboard.press("Shift+Tab")
        assert page.locator("#start").evaluate("element => element === document.activeElement")
        focus_style = page.locator("#start").evaluate("""element => {
          const style = getComputedStyle(element);
          return {outline:style.outlineStyle, width:parseFloat(style.outlineWidth),
            shadow:style.boxShadow, colour:style.outlineColor};
        }""")
        assert focus_style["outline"] != "none" and focus_style["width"] > 0 or focus_style["shadow"] != "none", focus_style
        idle_path = args.report.parent / "desktop-idle.png"
        save_screenshot(page, idle_path)
        screenshots.append(str(idle_path.relative_to(ROOT)))

        # Detection-confidence adjustment remains keyboard accessible, away from routine operation.
        advanced = open_details_for(page, "#confidence")
        assert advanced.count(), "Advanced adjustment should have a compact disclosure"
        page.locator("#confidence").focus()
        page.keyboard.press("ArrowRight")
        assert page.locator("#confidenceValue").inner_text() == "55%"
        page.keyboard.press("ArrowLeft")
        assert page.locator("#confidenceValue").inner_text() == "50%"
        advanced.evaluate("element => element.open = false")

        page.locator("#start").click()
        page.wait_for_function("document.querySelector('#video').srcObject !== null", timeout=30000)
        page.wait_for_function("Number(document.querySelector('#latency').dataset.frameNumber) > 0", timeout=60000)
        assert page.locator("#start").is_disabled()
        assert not page.locator("#stop").is_disabled()
        assert "≈" not in page.locator("#detections").inner_text(), "A distance appeared before calibration"
        blue_colour = assert_detection_colour(page, "blue")
        assert re.search(r"\b[0-9]+%\b|\b[0-9]+%", page.locator("#detections").inner_text()), "Detection confidence is missing"

        # Setup stays next to the image, before routine readouts, on a phone.
        assert page.locator("#setupPanel").evaluate("element => element.open")
        for width, height in [(390,844), (320,740)]:
            page.set_viewport_size({"width":width,"height":height})
            page.evaluate("window.scrollTo(0, 0)")
            assert_no_overflow(page)
            key = f"mobile_{width}_initial_reference"
            layouts[key] = geometry(page)
            controls_bottom = page.locator("#calibrate").bounding_box()
            assert controls_bottom["y"] + controls_bottom["height"] <= height, (key, controls_bottom)
            assert layouts[key]["reference"]["y"] >= layouts[key]["viewer"]["bottom"] - 2, layouts[key]
            assert layouts[key]["inspector"]["y"] >= layouts[key]["reference"]["bottom"] - 2, layouts[key]
            path = args.report.parent / f"mobile_{width}-initial-reference.png"
            save_screenshot(page, path)
            screenshots.append(str(path.relative_to(ROOT)))
        page.set_viewport_size({"width":1440,"height":1000})

        calibration = open_details_for(page, "#referenceMetres")
        page.locator("#referenceMetres").fill("3.00")
        page.evaluate("""() => {
          window.__qaCalibrationMessages = [];
          const element = document.querySelector('#calibrationState');
          new MutationObserver(() => window.__qaCalibrationMessages.push(element.textContent))
            .observe(element, {childList:true,subtree:true,characterData:true});
        }""")
        page.locator("#calibrate").click()
        page.wait_for_function("document.querySelector('#calibrationState').textContent.includes('Camera reference saved')", timeout=60000)
        saved = page.evaluate("JSON.parse(localStorage.getItem('cube-lens-pro-reference-v1'))")
        assert saved["sideMetres"] == .35 and saved["referenceMetres"] == 3, saved
        assert not page.locator("#setupPanel").evaluate("element => element.open"), "Successful reference should return to compact operation"
        events = page.evaluate("window.__qaCalibrationMessages")
        observed_samples = sorted({int(value) for text in events for value in re.findall(r'Reference (\d+)/7', text)})
        assert len(observed_samples) >= 5, (observed_samples, events)
        page.wait_for_function("document.querySelector('#detections').textContent.includes('≈')", timeout=30000)
        metres = read_distance(page, "35 cm reference")
        assert abs(metres / 3 - 1) < .1, metres
        if calibration.count():
            calibration.evaluate("element => element.open = false")

        layouts["desktop_1440"] = geometry(page)
        assert layouts["desktop_1440"]["viewer"]["width"] >= 2 * layouts["desktop_1440"]["inspector"]["width"], layouts
        assert layouts["desktop_1440"]["viewer"]["right"] <= layouts["desktop_1440"]["inspector"]["x"] + 2, layouts
        assert_no_overflow(page)
        desktop_path = args.report.parent / "desktop-measuring.png"
        save_screenshot(page, desktop_path)
        screenshots.append(str(desktop_path.relative_to(ROOT)))

        previous = frame_number(page)
        page.evaluate("() => window.setScene('motion-8.jpg')")
        wait_frames(page, previous)
        red_colour = assert_detection_colour(page, "red")
        assert abs(read_distance(page, "red cube") / 3 - 1) < .1
        previous = frame_number(page)
        page.evaluate("() => window.setScene('motion-0.jpg')")
        wait_frames(page, previous)

        for width, height in [(390,844), (320,740)]:
            page.set_viewport_size({"width":width,"height":height})
            page.wait_for_timeout(200)
            key = f"mobile_{width}"
            layouts[key] = geometry(page)
            assert_no_overflow(page)
            assert layouts[key]["viewer"]["width"] >= width - 48, layouts[key]
            assert layouts[key]["inspector"]["y"] >= layouts[key]["viewer"]["bottom"] - 2, layouts[key]
            assert layouts[key]["inspector"]["y"] >= layouts[key]["reference"]["bottom"] - 2, layouts[key]
            assert page.locator("#video").is_visible()
            path = args.report.parent / f"{key}-measuring.png"
            save_screenshot(page, path)
            screenshots.append(str(path.relative_to(ROOT)))
        page.set_viewport_size({"width":390,"height":844})
        open_details_for(page, "#referenceMetres")
        calibration_path = args.report.parent / "mobile-390-calibration.png"
        save_screenshot(page, calibration_path)
        screenshots.append(str(calibration_path.relative_to(ROOT)))
        assert_no_overflow(page)

        page.locator("#stop").click()
        page.wait_for_function("document.querySelector('#video').srcObject === null && !document.querySelector('#start').disabled")
        stopped_frame = frame_number(page)
        page.wait_for_timeout(250)
        assert frame_number(page) == stopped_frame
        assert page.locator("#stop").is_disabled()
        assert "paused" in page.locator("#status").inner_text().lower()
        page.locator("#start").click()
        page.wait_for_function("document.querySelector('#detections').textContent.includes('≈')", timeout=30000)
        wait_frames(page, stopped_frame, 2)
        assert abs(read_distance(page, "restarted camera") / 3 - 1) < .1
        page.locator("#stop").click()
        page.wait_for_function("!document.querySelector('#photoFile').disabled")
        before_photo = frame_number(page)
        page.locator("#photoFile").set_input_files(ROOT / "tests/fixtures/blue.jpg")
        page.wait_for_function("document.querySelector('#status').textContent.includes('Photo processed')", timeout=30000)
        assert frame_number(page) > before_photo
        assert "blue" in page.locator("#detections").inner_text().lower()
        assert "≈" not in page.locator("#detections").inner_text(), "Photo inherited camera intrinsics"
        page.wait_for_function("!document.querySelector('#save').disabled")
        with page.expect_download() as image_download:
            page.locator("#save").click()
        image = image_download.value
        assert image.failure() is None and image.suggested_filename.endswith(".png")
        assert Path(image.path()).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        with page.expect_download() as diagnostics_download:
            page.locator("#diagnostics").click()
        diagnostics = json.loads(Path(diagnostics_download.value.path()).read_text())
        assert diagnostics["version"] == "cube-lens-pro-1" and diagnostics["frames"]
        assert diagnostics["reference"]["metres"] == 3
        assert "data:image" not in json.dumps(diagnostics)
        assert_no_overflow(page)
        photo_path = args.report.parent / "mobile-390-photo.png"
        save_screenshot(page, photo_path)
        screenshots.append(str(photo_path.relative_to(ROOT)))

        denied_context = browser.new_context(viewport={"width":390,"height":844}, service_workers="block")
        denied = denied_context.new_page()
        denied.on("pageerror", lambda error: errors.append(str(error)))
        denied.add_init_script("navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Denied by test', 'NotAllowedError'); };")
        denied.goto(args.url, wait_until="networkidle")
        denied.wait_for_function("document.querySelector('#modelState').textContent.includes('Ready')", timeout=60000)
        denied.locator("#start").click()
        denied.wait_for_function("document.querySelector('#status').textContent.includes('Camera permission denied')")
        assert not denied.locator("#start").is_disabled()
        assert denied.locator("#stop").is_disabled()
        assert_no_overflow(denied)
        denied_path = args.report.parent / "mobile-390-permission-error.png"
        save_screenshot(denied, denied_path)
        screenshots.append(str(denied_path.relative_to(ROOT)))
        denied_context.close()
        assert not errors, errors
        browser.close()

    report = {"passed":True, "date":datetime.now(timezone.utc).isoformat(),
              "scope":"Instrument workstation UI regression using genuine browser inference and synthetic camera input",
              "checks":["camera dominates desktop; inspector at right", "mobile inspector below camera",
                        "1440, 390 and 320 pixel layouts without horizontal overflow",
                        "mobile initial reference controls fit first viewport; reference before inspector",
                        "successful calibration collapses to compact reference bar",
                        "idle readouts do not invent measurements", "visible keyboard focus",
                        "confidence disclosure and keyboard slider", "reference collects seven-frame setup",
                        "35 cm calibration and 3 m reading", "red/blue semantic detection colours",
                        "camera stop and restart", "photo classification without inherited camera distance",
                        "PNG export", "diagnostics export without image data", "recoverable permission denial"],
              "layouts":layouts, "focus_style":focus_style,
              "detection_colours":{"blue":blue_colour,"red":red_colour},
              "reference_progress_observed":observed_samples, "calibrated_synthetic_metres":metres,
              "diagnostics_frame_count":len(diagnostics["frames"]), "screenshots":screenshots,
              "limitations":"Desktop Chromium and synthetic cube geometry. No physical accuracy or iQOO Neo 6 timing claim."}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
