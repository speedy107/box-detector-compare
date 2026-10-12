"""Browser smoke checks; supply a compatible model and known blue/red test photos."""
import argparse
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8001/')
parser.add_argument('--model', required=True)
parser.add_argument('--blue', required=True, help='Photo with exactly one blue box')
parser.add_argument('--red', required=True, help='Photo with exactly one red box')
parser.add_argument('--chromium', help='Optional installed Chromium executable')
args = parser.parse_args()

with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=args.chromium,
        headless=True,
        args=['--no-sandbox', '--use-fake-device-for-media-stream',
              '--use-fake-ui-for-media-stream'],
    )
    context = browser.new_context(
        viewport={'width': 1360, 'height': 1000},
        permissions=['camera'], accept_downloads=True,
    )
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(args.url, wait_until='networkidle')
    page.locator('#photoFile').set_input_files(args.blue)
    page.wait_for_function(
        "document.querySelector('#status').textContent.startsWith('V5 analyzed')",
        timeout=60000,
    )
    assert page.locator('#v5Counts').inner_text() == '1 blue · 0 red'
    page.locator('#modelFile').set_input_files(args.model)
    page.wait_for_function(
        "document.querySelector('#status').textContent.startsWith('Comparison complete')",
        timeout=60000,
    )
    assert page.locator('#uploadedCounts').inner_text() == '1 blue · 0 red'
    page.locator('#photoFile').set_input_files(args.red)
    page.wait_for_function(
        "document.querySelector('#status').textContent.startsWith('Comparison complete')",
        timeout=60000,
    )
    assert page.locator('#v5Counts').inner_text() == '0 blue · 1 red'
    assert page.locator('#uploadedCounts').inner_text() == '0 blue · 1 red'
    assert page.locator('#agreement').inner_text().startswith('1 matched')
    with page.expect_download() as download:
        page.locator('#save').click()
    export = Path(tmp) / 'comparison.png'
    download.value.save_as(export)
    assert export.stat().st_size > 1000
    page.locator('#confidence').fill('50')
    page.locator('#confidence').dispatch_event('change')
    page.wait_for_function(
        "document.querySelector('#confidenceValue').textContent==='50%' "
        "&& !document.querySelector('#save').disabled"
    )
    invalid = Path(tmp) / 'invalid.onnx'
    invalid.write_bytes(b'not an ONNX model')
    page.locator('#modelFile').set_input_files(str(invalid))
    page.wait_for_function(
        "document.querySelector('#status').textContent.startsWith('Could not load this model')",
        timeout=60000,
    )
    assert Path(args.model).name in page.locator('#modelName').inner_text()
    page.locator('#photoFile').set_input_files(args.red)
    page.wait_for_function(
        "document.querySelector('#status').textContent.startsWith('Comparison complete')",
        timeout=60000,
    )
    assert page.locator('#uploadedCounts').inner_text() == '0 blue · 1 red'
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.locator('#start').click()
    page.wait_for_function(
        "document.querySelector('#status').textContent.startsWith('Live comparison')",
        timeout=60000,
    )
    page.locator('#stop').click()
    page.wait_for_function("!document.querySelector('#start').disabled", timeout=60000)
    assert page.evaluate("document.querySelector('#video').srcObject===null")
    assert not errors, errors
    browser.close()
    print('PASS: V5 and uploaded inference, agreement, export, confidence, invalid model recovery, mobile layout, simulated camera, stop.')
