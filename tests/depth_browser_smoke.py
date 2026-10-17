"""Exercise real YOLO + metric depth. Requires the verified exported depth artifact."""
import argparse
import json
from pathlib import Path
import re
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8001/')
parser.add_argument('--blue',required=True)
parser.add_argument('--red',required=True)
parser.add_argument('--chromium')
parser.add_argument('--camera-video',type=Path,default=Path(__file__).resolve().parent/'fixtures/blue.y4m')
parser.add_argument('--report',type=Path)
args=parser.parse_args()
assert (Path(__file__).resolve().parents[1]/'models/depth-anything-v2-metric-indoor-q8.onnx').is_file(), 'Export the real, verified metric-depth model first'

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=args.chromium,headless=True,args=['--no-sandbox','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',f'--use-file-for-fake-video-capture={args.camera_video.resolve()}'])
    context=browser.new_context(viewport={'width':390,'height':844},permissions=['camera'])
    page=context.new_page();errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    observations=[]
    page.goto(args.url,wait_until='networkidle')
    page.wait_for_function("document.querySelector('#modelState').textContent==='Ready'",timeout=60000)
    assert page.locator('#depthEnabled').is_checked()
    page.locator('#distanceMode').select_option('ai')
    for image,expected in [(args.blue,'1 blue · 0 red'),(args.red,'0 blue · 1 red')]:
        page.locator('#photoFile').set_input_files(image)
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('Photo detection complete')",timeout=180000)
        assert page.locator('#counts').inner_text()==expected
        text=page.locator('#detectionList').inner_text()
        distance=re.search(r'≈ ([0-9]+\.[0-9]+) m',text)
        assert distance and 0<float(distance.group(1))<20,text
        assert page.locator('#depthState').inner_text()=='Indoor depth ready'
        assert 'depth' in page.locator('#timing').inner_text()
        observations.append({'image':Path(image).name,'detections':text,'timing':page.locator('#timing').inner_text()})
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    page.locator('#depthEnabled').uncheck()
    page.wait_for_function("!document.querySelector('#save').disabled")
    assert ' m' not in page.locator('#detectionList').inner_text()
    page.locator('#depthEnabled').check()
    page.wait_for_function("!document.querySelector('#save').disabled")
    assert '≈' in page.locator('#detectionList').inner_text()
    # A failed depth download must preserve honest detection-only results.
    failure=browser.new_context()
    failure.route('**/models/depth-anything-v2-metric-indoor-q8.onnx',lambda route:route.abort())
    other=failure.new_page();other.goto(args.url,wait_until='networkidle')
    other.wait_for_function("document.querySelector('#modelState').textContent==='Ready'",timeout=60000)
    other.locator('#distanceMode').select_option('ai')
    other.locator('#photoFile').set_input_files(args.blue)
    other.wait_for_function("document.querySelector('#status').textContent.startsWith('Detection works, but distance is unavailable')",timeout=60000)
    assert other.locator('#counts').inner_text()=='1 blue · 0 red'
    assert 'Depth unavailable' in other.locator('#detectionList').inner_text()
    assert '≈' not in other.locator('#detectionList').inner_text()
    assert other.locator('#depthRetry').is_visible()
    failure.unroute('**/models/depth-anything-v2-metric-indoor-q8.onnx')
    other.locator('#depthRetry').click()
    other.wait_for_function("document.querySelector('#depthState').textContent==='Indoor depth ready' && !document.querySelector('#save').disabled",timeout=180000)
    assert '≈' in other.locator('#detectionList').inner_text()
    failure.close()

    # Reject an artifact whose bytes do not match its metadata, then recover.
    integrity=browser.new_context()
    def corrupt_checksum(route):
        response=route.fetch();metadata=response.json();metadata['onnx_sha256']='0'*64
        route.fulfill(response=response,json=metadata)
    integrity.route('**/models/depth-anything-v2-metric-indoor-q8.json',corrupt_checksum)
    tampered=integrity.new_page();tampered.goto(args.url,wait_until='networkidle')
    tampered.wait_for_function("document.querySelector('#modelState').textContent==='Ready'",timeout=60000)
    tampered.locator('#distanceMode').select_option('ai')
    tampered.locator('#photoFile').set_input_files(args.blue)
    tampered.wait_for_function("document.querySelector('#status').textContent.includes('integrity check failed')",timeout=60000)
    assert tampered.locator('#counts').inner_text()=='1 blue · 0 red'
    assert '≈' not in tampered.locator('#detectionList').inner_text()
    integrity.unroute('**/models/depth-anything-v2-metric-indoor-q8.json')
    tampered.locator('#depthRetry').click()
    tampered.wait_for_function("document.querySelector('#depthState').textContent==='Indoor depth ready' && !document.querySelector('#save').disabled",timeout=180000)
    integrity.close()

    # The camera really sees a blue block, runs depth on it, and can stop mid-load.
    camera=browser.new_context(permissions=['camera'])
    camera_page=camera.new_page();camera_page.goto(args.url,wait_until='networkidle')
    camera_page.wait_for_function("document.querySelector('#modelState').textContent==='Ready'",timeout=60000)
    camera_page.locator('#distanceMode').select_option('ai')
    camera_page.locator('#start').click()
    camera_page.wait_for_function("document.querySelector('#status').textContent.startsWith('Live detection') && document.querySelector('#depthState').textContent==='Indoor depth ready'",timeout=180000)
    assert '≈' in camera_page.locator('#detectionList').inner_text()
    observations.append({'image':'simulated camera blue block','detections':camera_page.locator('#detectionList').inner_text(),'timing':camera_page.locator('#timing').inner_text()})
    camera_page.locator('#stop').click()
    camera_page.wait_for_function("!document.querySelector('#start').disabled",timeout=10000)
    assert camera_page.evaluate("document.querySelector('#video').srcObject===null")
    camera.close()
    stopping=browser.new_context(permissions=['camera'])
    stopped=stopping.new_page();stopped.goto(args.url,wait_until='networkidle')
    stopped.wait_for_function("document.querySelector('#modelState').textContent==='Ready'",timeout=60000)
    stopped.locator('#distanceMode').select_option('ai')
    # Hold the download until the worker is terminated by Stop.
    held_routes=[]
    stopping.route('**/models/depth-anything-v2-metric-indoor-q8.onnx',lambda route:held_routes.append(route))
    stopped.locator('#start').click()
    stopped.wait_for_function("document.querySelector('#depthState').textContent.startsWith('Loading indoor depth model')",timeout=60000)
    stopped.locator('#stop').click()
    stopped.wait_for_function("!document.querySelector('#start').disabled",timeout=10000)
    assert stopped.evaluate("document.querySelector('#video').srcObject===null")
    assert stopped.locator('#depthState').inner_text()=='Ready to load'
    stopped.locator('#depthEnabled').uncheck()
    stopped.locator('#photoFile').set_input_files(args.red)
    stopped.wait_for_function("document.querySelector('#status').textContent.startsWith('Photo detection complete')",timeout=60000)
    assert stopped.locator('#counts').inner_text()=='0 blue · 1 red'
    for route in held_routes:
        route.abort()
    stopping.unroute_all(behavior='ignoreErrors')
    stopping.close()
    assert not errors,errors
    browser.close()
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps({'passed':True,'browser':'Chromium','observations':observations,'checks':['real metric-depth model','per-block metres','mobile layout','toggle','failed download fallback and retry','checksum rejection and retry','camera with real depth inference','stop cancels loading worker','detection after cancellation'],'limitations':'Composite fixtures and simulated camera; no physical distance calibration or phone performance measurements.'},indent=2)+'\n')
    print('PASS: real metric depth, metres, mobile layout, toggle, download fallback/retry, integrity/retry, camera depth, stop/cancellation recovery.')
