"""Real YOLO camera inference + reference geometry. No physical accuracy claim."""
import argparse
import json
from pathlib import Path
import statistics
import re
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8002/box-detector-compare-pages/')
parser.add_argument('--chromium',default='/usr/bin/chromium')
parser.add_argument('--report',type=Path,default=Path('validation/calibrated-browser.json'))
args=parser.parse_args()
fixtures=Path(__file__).resolve().parent/'fixtures'

def ready(page):
    page.wait_for_function("document.querySelector('#modelState').textContent==='Ready'",timeout=60000)
def start(page):
    page.locator('#start').click()
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Live detection')",timeout=60000)
def samples(page,count=20):
    return page.evaluate("""count=>new Promise((resolve,reject)=>{
      const values=[];let previous='';
      const timing=document.querySelector('#timing');
      const timeout=setTimeout(()=>{observer.disconnect();reject(new Error('Frame timing sample timeout'));},45000);
      const observer=new MutationObserver(()=>{
        if(timing.dataset.totalMs===previous)return;previous=timing.dataset.totalMs;
        values.push({update_ms:Number(previous),distance_ms:Number(timing.dataset.distanceMs)});
        if(values.length===count){clearTimeout(timeout);observer.disconnect();resolve(values);}
      });observer.observe(timing,{attributes:true});
    })""",count)

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=args.chromium,headless=True,args=[
        '--no-sandbox','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',
        f'--use-file-for-fake-video-capture={fixtures/"blue.y4m"}'])
    context=browser.new_context(viewport={'width':390,'height':844},permissions=['camera'],accept_downloads=True)
    page=context.new_page();errors=[];requests=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.on('request',lambda request:requests.append(request.url))
    page.goto(args.url,wait_until='networkidle');ready(page)
    assert page.locator('#distanceMode').input_value()=='calibrated'
    start(page)
    assert 'Calibrate first' in page.locator('#detectionList').inner_text()
    assert '≈' not in page.locator('#detectionList').inner_text()
    page.locator('#referenceMetres').fill('0');page.locator('#calibrate').click()
    assert 'between 0.1 and 30' in page.locator('#calibrationState').inner_text()
    page.locator('#referenceMetres').fill('3.00');page.locator('#calibrate').click()
    page.wait_for_function("document.querySelector('#calibrationState').textContent.includes('Reference saved at 3.00 m')",timeout=15000)
    page.wait_for_function("document.querySelector('#detectionList').textContent.includes('≈ 3.00 m')")
    normal=samples(page)
    assert all(v['update_ms']<1000 for v in normal),normal
    assert all(v['distance_ms']<10 for v in normal),normal
    cdp=context.new_cdp_session(page);cdp.send('Emulation.setCPUThrottlingRate',{'rate':4})
    throttled=samples(page)
    cdp.send('Emulation.setCPUThrottlingRate',{'rate':1})
    assert all(v['update_ms']<1000 for v in throttled),throttled
    assert not any('/models/depth-' in url for url in requests),'Fast mode must not load a depth model'
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    page.locator('#stop').click()
    page.wait_for_function("!document.querySelector('#save').disabled")
    args.report.parent.mkdir(parents=True,exist_ok=True)
    page.screenshot(path=str(args.report.with_suffix('.png')),full_page=True)
    with page.expect_download() as download:page.locator('#save').click()
    download.value.save_as(str(args.report.parent/'calibrated-detection.png'))
    start(page)
    page.evaluate("""()=>{
      const track=document.querySelector('#video').srcObject.getVideoTracks()[0];
      window.originalSettings=track.getSettings.bind(track);
      track.getSettings=()=>({...window.originalSettings(),zoom:2});
    }""")
    page.wait_for_function("document.querySelector('#depthState').textContent==='Recalibrate camera'")
    assert '≈' not in page.locator('#detectionList').inner_text()
    page.reload(wait_until='networkidle');ready(page);start(page)
    page.wait_for_function("document.querySelector('#detectionList').textContent.includes('≈ 3.00 m')",timeout=15000)
    # A camera calibration cannot be reused on an unrelated photo.
    page.locator('#stop').click();page.wait_for_function("!document.querySelector('#photoFile').disabled")
    page.locator('#photoFile').set_input_files(str(fixtures/'blue.jpg'))
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Photo detection complete')",timeout=60000)
    assert 'Use the calibrated camera' in page.locator('#detectionList').inner_text()
    assert '≈' not in page.locator('#detectionList').inner_text()
    # A user can switch out of a pending slow-model download without waiting for it.
    held=[]
    context.route('**/models/depth-anything-v2-metric-indoor-q8.onnx',lambda route:held.append(route))
    page.locator('#distanceMode').select_option('ai')
    page.wait_for_function("document.querySelector('#depthState').textContent.startsWith('Loading indoor depth model')",timeout=10000)
    page.locator('#distanceMode').select_option('calibrated')
    page.wait_for_function("!document.querySelector('#save').disabled",timeout=10000)
    assert '≈' not in page.locator('#detectionList').inner_text()
    for route in held:route.abort()
    context.unroute_all(behavior='ignoreErrors')
    start(page)
    page.locator('#clearCalibration').click()
    page.wait_for_function("document.querySelector('#detectionList').textContent.includes('Calibrate first')")
    assert '≈' not in page.locator('#detectionList').inner_text()
    assert page.evaluate("localStorage.getItem('box-lens-reference-v1')===null")
    page.locator('#stop').click()
    # Synthetic projection checks exercise real detector outlines at new scales.
    replay_context=browser.new_context(permissions=['camera'])
    replay=replay_context.new_page();replay.on('pageerror',lambda error:errors.append(str(error)))
    replay.add_init_script("""navigator.mediaDevices.getUserMedia=async()=>{
      const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
      window.setScene=async name=>{
        const response=await fetch(new URL('tests/fixtures/'+name,location.href));
        const image=await createImageBitmap(await response.blob());canvas.getContext('2d').drawImage(image,0,0);image.close();
      };
      await window.setScene('blue.jpg');const stream=canvas.captureStream(10),track=stream.getVideoTracks()[0];
      track.getSettings=()=>({deviceId:'geometry-replay',width:640,height:480,zoom:1,facingMode:'environment'});
      return stream;
    };""")
    replay.goto(args.url,wait_until='networkidle');ready(replay);start(replay)
    replay.locator('#referenceMetres').fill('3.00');replay.locator('#calibrate').click()
    replay.wait_for_function("document.querySelector('#calibrationState').textContent.includes('Reference saved')",timeout=15000)
    projection=[]
    for name,expected in [('blue-far.jpg',6),('blue-near.jpg',2)]:
        replay.evaluate('name=>window.setScene(name)',name)
        try:
            replay.wait_for_function("""expected=>{
              const found=document.querySelector('#detectionList').textContent.match(/≈ ([0-9.]+) m/);
              return found&&Math.abs(Number(found[1])/expected-1)<.12;
            }""",arg=expected,timeout=15000)
        except Exception as error:
            raise AssertionError(f'{name}: expected projection near {expected} m; {replay.locator("#detectionList").inner_text()}') from error
        observed=float(re.search(r'≈ ([0-9.]+) m',replay.locator('#detectionList').inner_text()).group(1))
        projection.append({'fixture':name,'expected_from_projection_metres':expected,'observed_metres':observed})
    replay_context.close()
    assert not errors,errors
    browser.close()

def summary(values):
    update=sorted(v['update_ms'] for v in values)
    return {'frames':len(update),'median_update_ms':statistics.median(update),
            'p95_update_ms':update[int(.95*(len(update)-1))],'max_update_ms':max(update),
            'max_distance_ms':max(v['distance_ms'] for v in values)}
report={'passed':True,'date':'2026-10-17','runtime':'Headless Chromium in cloud container; WASM, one CPU thread',
        'normal':summary(normal),'cpu_throttled_4x':summary(throttled),
        'synthetic_projection_checks':projection,
        'checks':['real YOLO on simulated camera','no invented distance before calibration','five-frame reference','no depth model requested in fast mode',
                  'mobile layout','save annotated detection','zoom mismatch suppresses distance','reference survives reload with matching camera',
                  'photos cannot reuse camera calibration','cancel slow AI by switching modes','clear reference suppresses distance'],
        'limitations':'Composite fixtures and simulated camera. These are container timings, not iQOO Neo 6 measurements. Geometry checks are not physical accuracy validation.'}
args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
