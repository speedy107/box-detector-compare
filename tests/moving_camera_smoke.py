"""Real YOLO + OpenCV pose on rendered 35 cm cubes at known viewpoints."""
import argparse
import json
from pathlib import Path
import re
import statistics
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8002/box-detector-compare-pages/')
parser.add_argument('--chromium',default='/usr/bin/chromium')
parser.add_argument('--report',type=Path,default=Path('validation/moving-camera-browser.json'))
args=parser.parse_args()
scenes=json.loads((Path(__file__).resolve().parent/'fixtures/motion/scenes.json').read_text())

def sample_frames(page,count=20):
    return page.evaluate("""count=>new Promise((resolve,reject)=>{
      const values=[];let previous='';const timing=document.querySelector('#timing');
      const timeout=setTimeout(()=>{observer.disconnect();reject(new Error('Timing sample timeout'));},45000);
      const observer=new MutationObserver(()=>{
        if(timing.dataset.frameNumber===previous)return;previous=timing.dataset.frameNumber;
        values.push({update_ms:Number(timing.dataset.totalMs),pose_ms:Number(timing.dataset.distanceMs)});
        if(values.length===count){clearTimeout(timeout);observer.disconnect();resolve(values);}
      });observer.observe(timing,{attributes:true});
    })""",count)

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=args.chromium,headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':390,'height':844},permissions=['camera'])
    page=context.new_page();errors=[];requests=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.on('request',lambda request:requests.append(request.url))
    page.add_init_script("""navigator.mediaDevices.getUserMedia=async()=>{
      const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
      window.setScene=async name=>{
        const image=await createImageBitmap(await (await fetch(new URL('tests/fixtures/motion/'+name,location.href))).blob());
        canvas.getContext('2d').drawImage(image,0,0);image.close();
        const video=document.querySelector('#video');
        if(video.srcObject)await new Promise(resolve=>video.requestVideoFrameCallback(resolve));
      };
      await window.setScene('motion-0.jpg');const stream=canvas.captureStream(10),track=stream.getVideoTracks()[0];
      track.getSettings=()=>({deviceId:'moving-camera-scene',width:640,height:480,zoom:1,facingMode:'environment'});
      return stream;
    };""")
    page.goto(args.url,wait_until='networkidle')
    page.wait_for_function("document.querySelector('#modelState').textContent==='Ready'",timeout=60000)
    assert page.locator('#distanceMode').input_value()=='pose'
    page.locator('#start').click()
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Live detection')",timeout=60000)
    assert 'Set camera reference' in page.locator('#detectionList').inner_text()
    assert '≈' not in page.locator('#detectionList').inner_text()
    page.locator('#referenceMetres').fill('3.00');page.locator('#calibrate').click()
    page.wait_for_function("document.querySelector('#calibrationState').textContent.includes('Camera reference saved') || (!document.querySelector('#calibrate').textContent.includes('Cancel') && document.querySelector('#calibrationState').textContent.includes('clear'))",timeout=60000)
    assert 'Camera reference saved' in page.locator('#calibrationState').inner_text(),page.locator('#calibrationState').inner_text()
    page.wait_for_function("document.querySelector('#detectionList').textContent.includes('≈')")
    observations=[]
    for scene in scenes:
        before=int(page.locator('#timing').get_attribute('data-frame-number'))
        page.evaluate('name=>window.setScene(name)',scene['name'])
        page.wait_for_function("before=>Number(document.querySelector('#timing').dataset.frameNumber)>=before+2",arg=before,timeout=15000)
        text=page.locator('#detectionList').inner_text();found=re.search(r'≈ ([0-9.]+) m',text)
        assert found,f'{scene["name"]}: {text}'
        metres=float(found.group(1));error=abs(metres/scene['truth_metres']-1)
        assert error<.1,(scene['name'],scene['angles'],scene['truth_metres'],metres)
        assert 'reference angle' not in text.lower()
        observations.append({'scene':scene['name'],'camera_angles_degrees':scene['angles'],'synthetic_surface_metres':scene['truth_metres'],
                             'estimated_metres':metres,'relative_error':error,'update_ms':float(page.locator('#timing').get_attribute('data-total-ms'))})
    normal=sample_frames(page)
    cdp=context.new_cdp_session(page);cdp.send('Emulation.setCPUThrottlingRate',{'rate':4})
    slowed=sample_frames(page);cdp.send('Emulation.setCPUThrottlingRate',{'rate':1})
    assert all(v['update_ms']<1000 for v in normal),normal
    assert all(v['update_ms']<1000 for v in slowed),slowed
    assert not any('/models/depth-' in url for url in requests)
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    page.locator('#stop').click();page.wait_for_function("!document.querySelector('#start').disabled")
    args.report.parent.mkdir(parents=True,exist_ok=True)
    page.screenshot(path=str(args.report.with_suffix('.png')),full_page=True)
    assert page.evaluate("JSON.parse(localStorage.getItem('box-lens-cube-camera-v1')).sideMetres===.35")
    page.locator('#start').click();page.wait_for_function("document.querySelector('#video').srcObject!==null && document.querySelector('#status').textContent.startsWith('Live detection')")
    page.wait_for_function("document.querySelector('#detectionList').textContent.includes('≈')")
    page.evaluate("""()=>{
      const track=document.querySelector('#video').srcObject.getVideoTracks()[0],get=track.getSettings.bind(track);
      track.getSettings=()=>({...get(),zoom:2});
    }""")
    page.wait_for_function("document.querySelector('#depthState').textContent==='Recalibrate camera'")
    assert '≈' not in page.locator('#detectionList').inner_text()
    page.locator('#clearCalibration').click()
    page.wait_for_function("document.querySelector('#detectionList').textContent.includes('Set camera reference')")
    assert page.evaluate("localStorage.getItem('box-lens-cube-camera-v1')===null")
    assert not errors,errors
    browser.close()

def summary(values):
    update=sorted(v['update_ms'] for v in values)
    return {'frames':len(values),'median_update_ms':statistics.median(update),'p95_update_ms':update[int(.95*(len(update)-1))],
            'maximum_update_ms':max(update),'maximum_pose_ms':max(v['pose_ms'] for v in values)}
report={'passed':True,'date':'2026-10-17','runtime':'Headless Chromium container; single-thread WASM; real YOLO and OpenCV cube pose',
        'normal':summary(normal),'chrome_cpu_slowdown_4x':summary(slowed),'viewpoints':observations,
        'checks':['moving-camera mode by default','one five-frame camera reference','changed yaw/pitch/roll without re-referencing',
                  'changed distance','no AI depth model requested','mobile layout','zoom change invalidates camera reference','clear reference'],
        'limitations':'Rendered cubes and a simulated camera. Not measured on iQOO Neo 6 and not a physical accuracy validation. Clear full cube corners and accurate 35 cm dimensions are required.'}
args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
