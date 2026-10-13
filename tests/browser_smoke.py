"""Verify the bundled best.onnx detector with known blue/red test photos."""
import argparse
import hashlib
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8001/')
parser.add_argument('--blue', required=True, help='Photo with exactly one blue box')
parser.add_argument('--red', required=True, help='Photo with exactly one red box')
parser.add_argument('--negative', help='Optional photo containing no boxes')
parser.add_argument('--chromium', help='Optional installed Chromium executable')
args = parser.parse_args()

with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=args.chromium, headless=True,
        args=['--no-sandbox', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
    )
    context = browser.new_context(viewport={'width': 1360, 'height': 1000},
                                  permissions=['camera'], accept_downloads=True)
    page = context.new_page()
    errors, model_responses, requests = [], [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('request', lambda request: requests.append(request.url))
    page.on('response', lambda response: model_responses.append(response)
            if response.url.endswith('/models/best.onnx') else None)
    page.goto(args.url, wait_until='networkidle')
    page.wait_for_function("document.querySelector('#modelState').textContent==='Ready'", timeout=60000)
    assert page.locator('#modelFile').count() == 0
    assert page.locator('canvas').count() == 1
    assert len(model_responses) == 1
    served_hash = hashlib.sha256(model_responses[0].body()).digest()
    bundled_hash = hashlib.sha256((Path(__file__).resolve().parents[1] / 'models/best.onnx').read_bytes()).digest()
    assert served_hash == bundled_hash
    assert not any('boxes-yolo11n-v5.onnx' in url for url in requests)
    page.locator('#depthEnabled').uncheck()
    for photo, expected in [(args.blue, '1 blue · 0 red'), (args.red, '0 blue · 1 red')]:
        page.locator('#photoFile').set_input_files(photo)
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('Photo detection complete')", timeout=60000)
        assert page.locator('#counts').inner_text() == expected
    with page.expect_download() as download:
        page.locator('#save').click()
    export = Path(tmp) / 'detection.png'
    download.value.save_as(export)
    assert export.stat().st_size > 1000
    page.locator('#confidence').fill('50')
    page.locator('#confidence').dispatch_event('change')
    page.wait_for_function("document.querySelector('#confidenceValue').textContent==='50%' && !document.querySelector('#save').disabled")
    if args.negative:
        page.locator('#photoFile').set_input_files(args.negative)
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('Photo detection complete')", timeout=60000)
        assert page.locator('#counts').inner_text() == 'No boxes detected'
    invalid = Path(tmp) / 'invalid.jpg'
    invalid.write_bytes(b'not an image')
    page.locator('#photoFile').set_input_files(str(invalid))
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Could not open this photo')")
    page.locator('#photoFile').set_input_files(args.red)
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Photo detection complete')", timeout=60000)
    assert page.locator('#counts').inner_text() == '0 blue · 1 red'
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.locator('#start').click()
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Live detection')", timeout=60000)
    page.locator('#stop').click()
    page.wait_for_function("!document.querySelector('#start').disabled", timeout=60000)
    assert page.evaluate("document.querySelector('#video').srcObject===null")
    assert not errors, errors
    browser.close()
    print('PASS: exact bundled YOLO model, automatic load, detection-only mode, blue/red detection, export, confidence, invalid photo recovery, mobile layout, simulated camera, stop.')
