import {INPUT_SIZE,CLASSES,COLORS,letterbox,rgbaToTensor,decode} from './detector.js';
import {DEPTH_SIZE,depthTensor,medianBlockDepth,distanceLabel,createDepthEstimator} from './depth.js?v=metric2';

const $=id=>document.getElementById(id);
const frame=document.createElement('canvas'),prep=document.createElement('canvas');
prep.width=prep.height=INPUT_SIZE;
const prepContext=prep.getContext('2d',{willReadFrequently:true});
const video=$('video'),canvas=$('resultCanvas');
const depthPrep=document.createElement('canvas');depthPrep.width=depthPrep.height=DEPTH_SIZE;
const depthContext=depthPrep.getContext('2d',{willReadFrequently:true});depthContext.imageSmoothingQuality='high';
let modelPromise=null,stream=null,live=false,busy=false,hasFrame=false,epoch=0,rerun=false,lastResult=null;
let depthFailed=false,depthBusy=false;
const status=(message,error=false)=>{$('status').textContent=message;$('status').classList.toggle('error',error);};
const estimator=createDepthEstimator(message=>{if(depthBusy){$('depthState').textContent=message;status(message);}});
function controls(){
  $('photoFile').disabled=busy;$('start').disabled=busy||live;
  $('stop').disabled=!live;$('save').disabled=busy||!lastResult;
  $('depthEnabled').disabled=busy;$('depthRetry').disabled=busy;
}
function lock(value){busy=value;controls();}
async function loadModel(){
  if(!modelPromise)modelPromise=(async()=>{
    if(!globalThis.ort)throw new Error('The inference runtime could not load. Reload this page.');
    ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
    ort.env.wasm.wasmPaths=new URL('./vendor/',location.href).href;
    $('modelState').textContent='Loading…';status('Loading your best.onnx model…');
    const response=await fetch(new URL('./models/best.onnx',location.href));
    if(!response.ok)throw new Error(`Model download failed (${response.status}).`);
    const session=await ort.InferenceSession.create(await response.arrayBuffer(),{executionProviders:['wasm'],graphOptimizationLevel:'all'});
    try{
      const tensor=new ort.Tensor('float32',new Float32Array(3*INPUT_SIZE*INPUT_SIZE).fill(114/255),[1,3,INPUT_SIZE,INPUT_SIZE]);let outputs;
      try{
        outputs=await session.run({[session.inputNames[0]]:tensor});const o=outputs[session.outputNames[0]];
        if(o.dims.length!==3||o.dims[0]!==1||o.dims[1]!==6)throw new Error('Unexpected blue/red detector output.');
      }finally{tensor.dispose();if(outputs)Object.values(outputs).forEach(o=>o.dispose());}
      $('modelState').textContent='Ready';return session;
    }catch(error){await session.release();throw error;}
  })().catch(error=>{modelPromise=null;$('modelState').textContent='Retry needed';throw error;});
  return modelPromise;
}
function capture(source,width,height){
  const scale=Math.min(1,1280/Math.max(width,height));
  frame.width=Math.max(1,Math.round(width*scale));frame.height=Math.max(1,Math.round(height*scale));
  frame.getContext('2d').drawImage(source,0,0,frame.width,frame.height);hasFrame=true;
}
function paint(result){
  canvas.width=frame.width;canvas.height=frame.height;canvas.hidden=false;
  canvas.parentElement.style.aspectRatio=`${frame.width}/${frame.height}`;
  canvas.parentElement.querySelector('.placeholder').hidden=true;
  const context=canvas.getContext('2d');context.drawImage(frame,0,0);
  if(!result)return;
  const font=Math.max(14,Math.round(frame.width*.025));context.font=`600 ${font}px system-ui`;context.lineWidth=Math.max(2,frame.width*.004);
  for(const box of result.boxes){
    context.strokeStyle=COLORS[box.classId];context.strokeRect(box.x1,box.y1,box.x2-box.x1,box.y2-box.y1);
    const text=`${CLASSES[box.classId]} ${Math.round(box.score*100)}%${result.depthEnabled?' · '+distanceLabel(box.distance):''}`,width=context.measureText(text).width+12;
    const x=Math.min(box.x1,Math.max(0,frame.width-width)),y=Math.max(0,box.y1-font-12);
    context.fillStyle=COLORS[box.classId];context.fillRect(x,y,width,font+12);context.fillStyle='#091321';context.fillText(text,x+6,y+font+2);
  }
  $('timing').textContent=`YOLO ${Math.round(result.elapsed)} ms${result.depthElapsed!==null?' · depth '+Math.round(result.depthElapsed)+' ms':''}`;
  $('counts').textContent=result.boxes.length?`${result.boxes.filter(b=>b.classId===0).length} blue · ${result.boxes.filter(b=>b.classId===1).length} red`:'No boxes detected';
  const list=$('detectionList');list.replaceChildren();
  for(const box of result.boxes){const item=document.createElement('li');item.textContent=`${CLASSES[box.classId]} ${Math.round(box.score*100)}%${result.depthEnabled?' · '+distanceLabel(box.distance):''}`;item.classList.toggle('red',box.classId===1);list.append(item);}
}
async function detect(){
  if(!hasFrame)return;
  if(busy){rerun=true;return;}
  lock(true);rerun=false;const token=epoch,confidence=Number($('confidence').value)/100;let tensor,outputs;
  try{
    const model=await loadModel(),geometry=letterbox(frame.width,frame.height);
    prepContext.fillStyle='rgb(114,114,114)';prepContext.fillRect(0,0,INPUT_SIZE,INPUT_SIZE);
    prepContext.drawImage(frame,geometry.left,geometry.top,geometry.resizedWidth,geometry.resizedHeight);
    tensor=new ort.Tensor('float32',rgbaToTensor(prepContext.getImageData(0,0,INPUT_SIZE,INPUT_SIZE).data),[1,3,INPUT_SIZE,INPUT_SIZE]);
    const start=performance.now();outputs=await model.run({[model.inputNames[0]]:tensor});
    const elapsed=performance.now()-start;
    const result={boxes:decode(outputs[model.outputNames[0]],geometry,confidence,.45),elapsed,depthElapsed:null,depthEnabled:$('depthEnabled').checked,depthError:null};
    result.boxes.forEach(box=>box.distance=null);
    // YOLO and depth consume this same captured frame; do not reuse a stale depth map.
    if(result.depthEnabled&&result.boxes.length&&!depthFailed){
      depthBusy=true;
      try{
        status('Estimating distance to the blocks…');
        depthContext.drawImage(frame,0,0,DEPTH_SIZE,DEPTH_SIZE);
        const depth=await estimator.infer(depthTensor(depthContext.getImageData(0,0,DEPTH_SIZE,DEPTH_SIZE).data));
        if(token!==epoch)return;
        result.depthElapsed=depth.elapsed;for(const box of result.boxes)box.distance=medianBlockDepth(box,depth,frame.width,frame.height);
        $('depthState').textContent='Indoor depth ready';
      }catch(error){
        if(token!==epoch)return;
        depthFailed=true;result.depthError=error.message;estimator.dispose();
        $('depthState').textContent='Depth unavailable';$('depthRetry').hidden=false;
      }finally{depthBusy=false;}
    }
    if(result.depthEnabled&&depthFailed)result.depthError=result.depthError||'Depth is unavailable. Retry depth to try again.';
    if(token!==epoch)return;
    lastResult=result;paint(result);
    status(result.depthError?`Detection works, but distance is unavailable: ${result.depthError}`:live?'Live detection · same-frame distance estimates · processing on this device':'Photo detection complete · processed on your device',Boolean(result.depthError));
  }catch(error){if(token===epoch){stopCamera();status(`Could not detect boxes: ${error.message}`,true);}}
  finally{tensor?.dispose();if(outputs)Object.values(outputs).forEach(o=>o.dispose());lock(false);if(rerun&&hasFrame&&!live){rerun=false;queueMicrotask(detect);}}
}
function stopCamera(message){
  live=false;epoch++;stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;controls();
  if(depthBusy){estimator.dispose();depthBusy=false;$('depthState').textContent='Ready to load';}
  if(message)status(message);
}
async function cameraLoop(token){
  if(!live||token!==epoch)return;
  if(!busy&&video.readyState>=2){capture(video,video.videoWidth,video.videoHeight);await detect();}
  if(live&&token===epoch)setTimeout(()=>cameraLoop(token),100);
}
$('photoFile').addEventListener('change',async event=>{
  const file=event.target.files[0];event.target.value='';if(!file||busy)return;
  stopCamera();lock(true);status('Opening your photo…');
  try{
    const bitmap=await createImageBitmap(file);try{capture(bitmap,bitmap.width,bitmap.height);}finally{bitmap.close();}
    lastResult=null;paint(null);$('counts').textContent='Analyzing…';$('timing').textContent='—';$('detectionList').replaceChildren();
  }catch(error){status(`Could not open this photo: ${error.message}`,true);lock(false);return;}
  lock(false);await detect();
});
$('start').addEventListener('click',async()=>{
  if(live||busy)return;
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia){status('Camera access requires HTTPS and a browser with camera support. You can still choose a photo.',true);return;}
  live=true;const token=++epoch;lock(true);status('Allow camera access to begin detection…');
  try{
    const camera=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}}});
    if(!live||token!==epoch){camera.getTracks().forEach(t=>t.stop());return;}
    stream=camera;video.srcObject=camera;await video.play();await loadModel();
    if(live&&token===epoch){lock(false);cameraLoop(token);}
  }catch(error){if(token===epoch){stopCamera();const messages={NotAllowedError:'Camera permission was denied. Allow it in browser settings or choose a photo.',NotFoundError:'No camera found. Choose a photo instead.',NotReadableError:'Camera is busy. Close other camera apps and try again.'};status(messages[error.name]||`Camera failed: ${error.message}`,true);}}
  finally{if(!live)lock(false);}
});
$('stop').addEventListener('click',()=>stopCamera('Camera stopped. Last detection remains on screen.'));
$('confidence').addEventListener('input',()=>{$('confidenceValue').textContent=`${$('confidence').value}%`;});
$('confidence').addEventListener('change',()=>{if(hasFrame&&!live)detect();});
$('depthEnabled').addEventListener('change',()=>{
  depthFailed=false;$('depthRetry').hidden=true;$('depthState').textContent=$('depthEnabled').checked?'Ready to load':'Distance off';
  if(hasFrame&&!live)detect();
});
$('depthRetry').addEventListener('click',()=>{depthFailed=false;estimator.dispose();$('depthRetry').hidden=true;$('depthState').textContent='Ready to load';if(hasFrame&&!live)detect();});
$('save').addEventListener('click',()=>{
  if(!lastResult||busy)return;
  canvas.toBlob(blob=>{if(!blob){status('Could not save this detection.',true);return;}const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`box-detection-${Date.now()}.png`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);},'image/png');
});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&live)stopCamera('Camera paused while this page was hidden. Start it again to continue.');});
window.addEventListener('pagehide',()=>{stopCamera();estimator.dispose();});
lock(true);
loadModel().then(()=>status('Your model is ready. Choose a photo or start the camera.')).catch(error=>status(`Could not load your model: ${error.message} Choose a photo or start camera to retry.`,true)).finally(()=>lock(false));
