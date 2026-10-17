import {INPUT_SIZE,CLASSES,COLORS,letterbox,rgbaToTensor,decode} from './detector.js';
import {DEPTH_SIZE,depthTensor,medianBlockDepth,distanceLabel,createDepthEstimator} from './depth.js?v=metric2';
import {REFERENCE_SAMPLES,cameraProfile,sameCamera,referenceSample,createCalibration,restoreCalibration,calibratedDistance} from './calibration.js?v=1';
import {restorePoseReference,createPoseEstimator,samePoseCamera} from './pose.js?v=2';

const $=id=>document.getElementById(id);
const frame=document.createElement('canvas'),prep=document.createElement('canvas');
prep.width=prep.height=INPUT_SIZE;
const prepContext=prep.getContext('2d',{willReadFrequently:true});
const video=$('video'),canvas=$('resultCanvas');
const depthPrep=document.createElement('canvas');depthPrep.width=depthPrep.height=DEPTH_SIZE;
const depthContext=depthPrep.getContext('2d',{willReadFrequently:true});depthContext.imageSmoothingQuality='high';
let modelPromise=null,stream=null,live=false,busy=false,hasFrame=false,epoch=0,rerun=false,lastResult=null;
let depthFailed=false,depthBusy=false;
let settingsRevision=0,frameProfile=null,capturedAt=0,calibrationRequest=null,calibration=null;
let poseReference=null,poseBusy=false,frameNumber=0;
const referenceKey='box-lens-reference-v1';
const poseReferenceKey='box-lens-cube-camera-v1';
try{calibration=restoreCalibration(JSON.parse(localStorage.getItem(referenceKey)));}catch{}
try{poseReference=restorePoseReference(JSON.parse(localStorage.getItem(poseReferenceKey)));}catch{}
const status=(message,error=false)=>{$('status').textContent=message;$('status').classList.toggle('error',error);};
const estimator=createDepthEstimator(message=>{if(depthBusy&&$('depthEnabled').checked&&$('distanceMode').value==='ai'){$('depthState').textContent=message;status(message);}});
const poseEstimator=createPoseEstimator(message=>{if(poseBusy&&$('distanceMode').value==='pose'){$('depthState').textContent=message;status(message);}});
function updateMethod(){
  const mode=$('distanceMode').value,pose=mode==='pose',fast=mode!=='ai',enabled=$('depthEnabled').checked;
  const reference=pose?poseReference:calibration;
  $('calibrationPanel').hidden=!fast||!enabled;
  $('methodBadge').textContent=enabled?(pose?'CUBE POSE':fast?'FIXED VIEW':'AI DEPTH'):'DETECTION';
  $('modeDescription').textContent=pose?'35 cm cubes · changing view angles · one camera reference':fast?'Legacy method · requires similar block angles':'Indoor AI estimate · can be inaccurate · slow on some phones';
  $('depthState').textContent=!enabled?'Distance off':fast?(reference?'Reference saved':'Reference needed'):'Ready to load';
  $('referenceHelp').textContent=pose?'Show one whole 35 × 35 × 35 cm cube. Enter the measured distance from the camera lens to its nearest surface. Hold at that distance during setup; your viewing angle may change afterwards.':'Show one whole, front-facing block near the centre. Enter the measured distance from the camera lens to its front face.';
  $('referenceCaveats').textContent=pose?'Use 35 cm cubes with visible, clear edges. Changing lens, zoom or stream resolution needs a new reference. Blurred, hidden or very small corners may prevent an estimate.':'Use equal-size blocks in the same orientation. Changing lens, zoom, camera orientation or resolution needs a new reference. Hidden edges and stacks can still cause errors.';
  $('referenceMetres').min=pose?'.3':'.1';$('referenceMetres').max=pose?'15':'30';
  if(!calibrationRequest)$('calibrationState').textContent=reference?(pose?`Camera reference saved at ${poseReference.referenceMetres.toFixed(2)} m. You may change viewing angle.`:`Reference saved at ${calibration.metres.toFixed(2)} m. Use the same camera settings and block orientation.`):'Start the camera and show one whole cube. No distance is guessed before setup.';
}
function controls(){
  $('photoFile').disabled=busy;$('start').disabled=busy||live;
  $('stop').disabled=!live;$('save').disabled=busy||!lastResult;
  // Distance controls stay usable while a slow AI inference is pending.
  $('depthRetry').disabled=busy;
  $('calibrate').disabled=!live||!$('depthEnabled').checked||$('distanceMode').value==='ai';
  $('calibrate').textContent=calibrationRequest?'Cancel reference':'Set reference';
  $('clearCalibration').disabled=!($('distanceMode').value==='pose'?poseReference:calibration)&&!calibrationRequest;
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
function capture(source,width,height,profile=null){
  capturedAt=performance.now();frameProfile=profile;frameNumber++;
  const scale=Math.min(1,1280/Math.max(width,height));
  frame.width=Math.max(1,Math.round(width*scale));frame.height=Math.max(1,Math.round(height*scale));
  frame.getContext('2d').drawImage(source,0,0,frame.width,frame.height);hasFrame=true;
}
function blockDistanceLabel(box,result){return result.mode!=='ai'?(box.distance===null?box.distanceReason:`≈ ${box.distance.toFixed(2)} m`):distanceLabel(box.distance);}
function paint(result){
  canvas.width=frame.width;canvas.height=frame.height;canvas.hidden=false;
  canvas.parentElement.style.aspectRatio=`${frame.width}/${frame.height}`;
  canvas.parentElement.querySelector('.placeholder').hidden=true;
  const context=canvas.getContext('2d');context.drawImage(frame,0,0);
  if(!result)return;
  const font=Math.max(14,Math.round(frame.width*.025));context.font=`600 ${font}px system-ui`;context.lineWidth=Math.max(2,frame.width*.004);
  for(const box of result.boxes){
    context.strokeStyle=COLORS[box.classId];context.strokeRect(box.x1,box.y1,box.x2-box.x1,box.y2-box.y1);
    const text=`${CLASSES[box.classId]} ${Math.round(box.score*100)}%${result.depthEnabled?' · '+blockDistanceLabel(box,result):''}`,width=context.measureText(text).width+12;
    const x=Math.min(box.x1,Math.max(0,frame.width-width)),y=Math.max(0,box.y1-font-12);
    context.fillStyle=COLORS[box.classId];context.fillRect(x,y,width,font+12);context.fillStyle='#091321';context.fillText(text,x+6,y+font+2);
    if(box.posePoints){context.fillStyle='#b2ffe1';for(const [px,py] of box.posePoints){context.beginPath();context.arc(px,py,Math.max(2,frame.width*.002),0,Math.PI*2);context.fill();}}
  }
  $('counts').textContent=result.boxes.length?`${result.boxes.filter(b=>b.classId===0).length} blue · ${result.boxes.filter(b=>b.classId===1).length} red`:'No boxes detected';
  const list=$('detectionList');list.replaceChildren();
  for(const box of result.boxes){const item=document.createElement('li');item.textContent=`${CLASSES[box.classId]} ${Math.round(box.score*100)}%${result.depthEnabled?' · '+blockDistanceLabel(box,result):''}`;item.classList.toggle('red',box.classId===1);list.append(item);}
  result.totalElapsed=performance.now()-result.startedAt;
  $('timing').textContent=`Update ${Math.round(result.totalElapsed)} ms · YOLO ${Math.round(result.elapsed)} ms${result.depthElapsed!==null?' · '+(result.mode==='pose'?'pose '+Math.round(result.depthElapsed):result.mode==='calibrated'?'distance '+result.depthElapsed.toFixed(1):'depth '+Math.round(result.depthElapsed))+' ms':''}`;
  $('timing').dataset.totalMs=String(result.totalElapsed);$('timing').dataset.distanceMs=String(result.depthElapsed??0);
  $('timing').dataset.frameNumber=String(result.frameNumber);
  $('latencyState').classList.toggle('slow',result.totalElapsed>=1000);
  $('latencyState').textContent=result.totalElapsed<1000?'This frame processed in under 1,000 ms.':'This frame exceeded 1,000 ms on this device.';
}
function collectReference(result){
  if(!calibrationRequest)return;
  try{
    if(!live||!frameProfile)throw new Error('Start the camera before setting a reference.');
    if(result.boxes.length!==1)throw new Error('Show exactly one complete block, then set the reference again.');
    const sample=referenceSample(result.boxes[0],frame.width,frame.height);
    if(sample.error)throw new Error(sample.error+'. Set the reference again.');
    const request=calibrationRequest;
    if(request.profile&&(!sameCamera(request.profile,frameProfile)||request.classId!==result.boxes[0].classId))
      throw new Error('Keep the same camera and block while setting the reference.');
    request.profile={...frameProfile};request.classId=result.boxes[0].classId;request.samples.push(sample);
    $('calibrationState').textContent=`Hold steady · reference frame ${request.samples.length}/${REFERENCE_SAMPLES}`;
    if(request.samples.length===REFERENCE_SAMPLES){
      calibration=createCalibration(request.samples,request.metres,request.profile);calibrationRequest=null;
      try{localStorage.setItem(referenceKey,JSON.stringify(calibration));}catch{}
      updateMethod();controls();
    }
  }catch(error){calibrationRequest=null;$('calibrationState').textContent=error.message;controls();}
}
function collectPoseReference(reference){
  if(!calibrationRequest||!reference)return;
  const request=calibrationRequest;
  if(request.profile&&!samePoseCamera(request.profile,frameProfile))throw new Error('Camera settings changed during setup. Try again.');
  request.profile={...frameProfile};request.samples.push(reference.focalNormalized*frame.width/Math.max(frame.width,frame.height));
  $('calibrationState').textContent=`Camera reference frame ${request.samples.length}/${REFERENCE_SAMPLES} · hold at the measured distance`;
  if(request.samples.length===REFERENCE_SAMPLES){
    const values=[...request.samples].sort((a,b)=>a-b),focalNormalized=values[Math.floor(values.length/2)];
    if(values.some(v=>Math.abs(v/focalNormalized-1)>.08))throw new Error('The camera reference was unstable. Hold at the measured distance and retry.');
    poseReference=restorePoseReference({version:1,focalBasis:'long-edge',sideMetres:.35,focalNormalized,referenceMetres:request.metres,profile:request.profile});
    if(!poseReference)throw new Error('This camera cannot safely save a reference.');
    calibrationRequest=null;try{localStorage.setItem(poseReferenceKey,JSON.stringify(poseReference));}catch{}
    updateMethod();controls();
  }
}
async function detect(){
  if(!hasFrame)return;
  if(busy){rerun=true;return;}
  lock(true);rerun=false;const token=epoch,revision=settingsRevision,confidence=Number($('confidence').value)/100;
  const mode=$('distanceMode').value,enabled=$('depthEnabled').checked,startedAt=live?capturedAt:performance.now();let tensor,outputs;
  try{
    const model=await loadModel(),geometry=letterbox(frame.width,frame.height);
    prepContext.fillStyle='rgb(114,114,114)';prepContext.fillRect(0,0,INPUT_SIZE,INPUT_SIZE);
    prepContext.drawImage(frame,geometry.left,geometry.top,geometry.resizedWidth,geometry.resizedHeight);
    tensor=new ort.Tensor('float32',rgbaToTensor(prepContext.getImageData(0,0,INPUT_SIZE,INPUT_SIZE).data),[1,3,INPUT_SIZE,INPUT_SIZE]);
    const start=performance.now();outputs=await model.run({[model.inputNames[0]]:tensor});
    const elapsed=performance.now()-start;
    if(token!==epoch||revision!==settingsRevision)return;
    const result={boxes:decode(outputs[model.outputNames[0]],geometry,confidence,.45),elapsed,startedAt,frameNumber,mode,depthElapsed:null,depthEnabled:enabled,depthError:null};
    result.boxes.forEach(box=>box.distance=null);
    if(enabled&&mode==='pose'){
      const matching=frameProfile&&poseReference&&samePoseCamera(poseReference.profile,frameProfile);
      result.boxes.forEach(box=>box.distanceReason=!frameProfile?'Use the calibrated camera':poseReference&&!matching?'Recalibrate camera':'Set camera reference');
      if(frameProfile&&(matching||calibrationRequest)&&result.boxes.length){
        poseBusy=true;
        try{
          const pixels=frame.getContext('2d',{willReadFrequently:true}).getImageData(0,0,frame.width,frame.height).data;
          const options=calibrationRequest?{referenceMetres:calibrationRequest.metres}:{focalNormalized:poseReference.focalNormalized*Math.max(frame.width,frame.height)/frame.width};
          const pose=await poseEstimator.infer(pixels,frame.width,frame.height,result.boxes,options);
          if(token!==epoch||revision!==settingsRevision)return;
          if(calibrationRequest){
            if(!pose.reference)throw new Error(pose.boxes[0]?.reason||'Show one complete cube with clear corners.');
            collectPoseReference(pose.reference);
          }
          result.depthElapsed=pose.elapsed;
          result.boxes.forEach((box,i)=>{
            box.posePoints=pose.boxes[i]?.points;
            if(!calibrationRequest&&poseReference&&samePoseCamera(poseReference.profile,frameProfile)){box.distance=pose.boxes[i]?.metres??null;box.distanceReason=pose.boxes[i]?.reason||'Cube corners unclear';}
          });
        }catch(error){
          if(token!==epoch||revision!==settingsRevision)return;
          if(calibrationRequest){calibrationRequest=null;$('calibrationState').textContent=error.message;controls();}
          result.boxes.forEach(box=>box.distanceReason=error.message);
        }finally{poseBusy=false;}
      }else if(calibrationRequest&&result.boxes.length===0){calibrationRequest=null;$('calibrationState').textContent='Show one complete cube, then set the camera reference again.';controls();}
      const currentMatches=frameProfile&&poseReference&&samePoseCamera(poseReference.profile,frameProfile);
      $('depthState').textContent=!frameProfile?'Use the calibrated camera':poseReference&&!currentMatches&&!calibrationRequest?'Recalibrate camera':poseReference?'Cube pose ready':'Reference needed';
    }
    if(enabled&&mode==='calibrated'){
      const distanceStart=performance.now();collectReference(result);
      for(const box of result.boxes){const estimate=calibratedDistance(box,frame.width,frame.height,calibration,frameProfile,result.boxes);box.distance=estimate.metres;box.distanceReason=estimate.reason;}
      result.depthElapsed=performance.now()-distanceStart;
      $('depthState').textContent=!calibration?'Reference needed':!frameProfile?'Use the calibrated camera':!sameCamera(calibration.profile,frameProfile)?'Recalibrate camera':'Fast calibrated';
    }
    // YOLO and depth consume this same captured frame; do not reuse a stale depth map.
    if(enabled&&mode==='ai'&&result.boxes.length&&!depthFailed){
      depthBusy=true;
      try{
        status('Estimating distance to the blocks…');
        depthContext.drawImage(frame,0,0,DEPTH_SIZE,DEPTH_SIZE);
        const depth=await estimator.infer(depthTensor(depthContext.getImageData(0,0,DEPTH_SIZE,DEPTH_SIZE).data));
        if(token!==epoch||revision!==settingsRevision)return;
        result.depthElapsed=depth.elapsed;for(const box of result.boxes)box.distance=medianBlockDepth(box,depth,frame.width,frame.height);
        $('depthState').textContent='Indoor depth ready';
      }catch(error){
        if(token!==epoch||revision!==settingsRevision)return;
        depthFailed=true;result.depthError=error.message;estimator.dispose();
        $('depthState').textContent='Depth unavailable';$('depthRetry').hidden=false;
      }finally{depthBusy=false;}
    }
    if(enabled&&mode==='ai'&&depthFailed)result.depthError=result.depthError||'Depth is unavailable. Retry depth to try again.';
    if(token!==epoch||revision!==settingsRevision)return;
    lastResult=result;paint(result);
    status(result.depthError?`Detection works, but distance is unavailable: ${result.depthError}`:live?`Live detection · ${enabled?(mode==='pose'?'moving-camera cube pose':mode==='calibrated'?'fixed-view distance':'AI distance estimate'):'distance off'} · processing on this device`:`Photo detection complete · ${enabled&&mode!=='ai'?'use Start camera for distance, or choose AI estimate':'processed on your device'}`,Boolean(result.depthError));
  }catch(error){if(token===epoch){stopCamera();status(`Could not detect boxes: ${error.message}`,true);}}
  finally{tensor?.dispose();if(outputs)Object.values(outputs).forEach(o=>o.dispose());lock(false);if(rerun&&hasFrame&&!live){rerun=false;queueMicrotask(detect);}}
}
function stopCamera(message){
  live=false;epoch++;if(calibrationRequest){calibrationRequest=null;$('calibrationState').textContent='Reference capture cancelled. Start the camera to try again.';}
  stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;controls();
  if(depthBusy){estimator.dispose();depthBusy=false;$('depthState').textContent='Ready to load';}
  if(poseBusy){poseEstimator.dispose();poseBusy=false;$('depthState').textContent=poseReference?'Reference saved':'Reference needed';}
  if(message)status(message);
}
async function cameraLoop(token){
  if(!live||token!==epoch)return;
  if(!busy&&video.readyState>=2){capture(video,video.videoWidth,video.videoHeight,cameraProfile(stream.getVideoTracks()[0].getSettings(),video.videoWidth,video.videoHeight));await detect();}
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
function distanceSettingsChanged(){
  settingsRevision++;calibrationRequest=null;depthFailed=false;
  if(depthBusy)estimator.dispose();
  if(poseBusy)poseEstimator.dispose();
  $('depthRetry').hidden=true;updateMethod();controls();
  if(hasFrame&&!live)detect();
}
$('depthEnabled').addEventListener('change',distanceSettingsChanged);
$('distanceMode').addEventListener('change',distanceSettingsChanged);
$('calibrate').addEventListener('click',()=>{
  if(calibrationRequest){calibrationRequest=null;$('calibrationState').textContent='Reference capture cancelled.';controls();return;}
  const metres=Number($('referenceMetres').value);
  const pose=$('distanceMode').value==='pose',minimum=pose ? .3 : .1,maximum=pose?15:30;
  if(!Number.isFinite(metres)||metres<minimum||metres>maximum){$('calibrationState').textContent=`Enter a measured distance between ${minimum} and ${maximum} metres.`;return;}
  if(!live)return;
  calibrationRequest={metres,samples:[],profile:null};$('calibrationState').textContent='Hold one complete block steady while the reference is captured.';controls();
});
$('clearCalibration').addEventListener('click',()=>{
  calibrationRequest=null;
  if($('distanceMode').value==='pose'){poseReference=null;try{localStorage.removeItem(poseReferenceKey);}catch{}}
  else{calibration=null;try{localStorage.removeItem(referenceKey);}catch{}}
  distanceSettingsChanged();
});
$('depthRetry').addEventListener('click',()=>{depthFailed=false;estimator.dispose();$('depthRetry').hidden=true;$('depthState').textContent='Ready to load';if(hasFrame&&!live)detect();});
$('save').addEventListener('click',()=>{
  if(!lastResult||busy)return;
  canvas.toBlob(blob=>{if(!blob){status('Could not save this detection.',true);return;}const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`box-detection-${Date.now()}.png`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);},'image/png');
});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&live)stopCamera('Camera paused while this page was hidden. Start it again to continue.');});
window.addEventListener('pagehide',()=>{stopCamera();estimator.dispose();poseEstimator.dispose();});
updateMethod();lock(true);
loadModel().then(()=>status('Your model is ready. Choose a photo or start the camera.')).catch(error=>status(`Could not load your model: ${error.message} Choose a photo or start camera to retry.`,true)).finally(()=>lock(false));
