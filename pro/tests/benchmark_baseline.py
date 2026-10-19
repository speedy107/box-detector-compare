"""Record the original site's processing time, cadence and UI stalls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from browser_smoke import ROOT, camera_script, sample_frames, timing_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8003/")
    parser.add_argument("--chromium", default="/usr/bin/chromium")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--report", type=Path, default=ROOT / "pro/validation/baseline.json")
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium, headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width":390,"height":844}, permissions=["camera"])
        page = context.new_page()
        page.add_init_script(camera_script("tests/fixtures/motion/"))
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function("document.querySelector('#modelState').textContent === 'Ready'", timeout=60000)
        page.locator("#start").click()
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('Live detection')", timeout=60000)
        page.locator("#referenceMetres").fill("3.00")
        page.locator("#calibrate").click()
        page.wait_for_function("document.querySelector('#calibrationState').textContent.includes('Camera reference saved')", timeout=60000)
        page.wait_for_function("document.querySelector('#detectionList').textContent.includes('≈')", timeout=30000)
        samples = sample_frames(page, selector="#timing", count=args.frames)
        page.locator("#stop").click()
        browser.close()
    report = {"date": datetime.now(timezone.utc).isoformat(), "variant": "Original site, main-thread YOLO and 100 ms inter-frame delay",
              "runtime": "Headless Chromium container; 30 fps controlled camera; synthetic 35 cm cube at 3 m",
              "timing": timing_summary(samples),
              "limitations": "Container benchmark only; no iQOO Neo 6 hardware measurement or physical accuracy claim."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report["timing"].items() if key != "raw"}, indent=2))


if __name__ == "__main__":
    main()
