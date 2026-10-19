import {createInferenceEngine} from './inference.js';
import {CLASSES,COLORS} from './detector.js';
import {REFERENCE_KEY,SAMPLE_COUNT,median,cameraProfile,sameCamera,restoreReference,finishReference} from './reference.js';

const $=id=>document.getElementById(id),video=$('video'),overlay=$('overlay'),photo=$('photoCanvas');
const overlayContext=overlay.getContext('2d');
const STALE_MS=750;
let live=false,stream=null,generation=0,revision=0,inflight=false,frameNumber=0,callbackId=null;
let videoSequence=0,capturedSequence=0,lastMediaTime=-1;
let reference=null,referenceRequest=null,ready=false,bootstrapComplete=false,backendInfo=null,lastResult=null,lastSnapshot=null,photoMode=false,wakeLock=null;
let spareCanvas=document.createElement('canvas'),lastPaintAt=0,lastCommitAt=0,stale=false,paintTick=0;
const history=[],intervals=[];
try{reference=restoreReference(JSON.parse(localStorage.getItem(REFERENCE_KEY)));}catch{}
const setText=(id,text)=>{if($(id).textContent!==text)$(id).textContent=text;};
const status=(text,error=false)=>{setText('status',text);$('status').classList.toggle('error',error);};
const engine=createInferenceEngine(message=>{if(!ready)status(message);});

function controls(){
  $('start').disabled=live||!bootstrapComplete;$('stop').disabled=!live;
  $('photoFile').disabled=!bootstrapComplete||inflight&&!live;
  $('save').disabled=!lastResult||!lastSnapshot;
  $('calibrate').disabled=!live||!ready;
  $('calibrate').textContent=referenceRequest?'Cancel reference':'Set camera reference';
  $('clearCalibration').disabled=!reference&&!referenceRequest;
  // Presentation only: expose the existing operating state to the interface.
  const cameraState=live?(stream?'live':'starting'):photoMode?'photo':lastSnapshot?'paused':'idle';
  document.body.dataset.cameraState=cameraState;
  document.body.dataset.referenceState=referenceRequest?'setting':reference?'stored':'needed';
  setText('cameraState',({live:'Live camera',starting:'Starting',photo:'Photo mode',paused:'Paused',idle:'Standby'})[cameraState]);
  setText('referenceBadge',referenceRequest?'Acquiring reference':reference?'Reference stored':'Unreferenced');
}
function referenceState(){
  setText('calibrationState',reference?`Camera reference saved at ${reference.referenceMetres.toFixed(2)} m. Move and change viewing angle freely.`:'Set one measured reference to enable distance readings.');
  setText('qualityState',reference?'Reference saved':'Reference needed');
  if(reference)$('setupPanel').open=false;
  controls();
}
async function warmup(){
  setText('modelState','Loading');
  try{
    await (window.cubeIsolationReady||Promise.resolve());bootstrapComplete=true;
    backendInfo=await engine.warmup();ready=true;setText('modelState','Ready');$('modelState').dataset.threads=String(backendInfo.threads);
    if(!live)status('Ready. Start your rear camera to begin.');controls();
  }
  catch(error){ready=false;setText('modelState','Retry needed');status(error.message,true);controls();throw error;}
}
function frameSize(width,height){const scale=Math.min(1,1280/Math.max(width,height));return {width:Math.round(width*scale),height:Math.round(height*scale)};}
function dimensions(width,height){
  if(overlay.width!==width||overlay.height!==height){overlay.width=width;overlay.height=height;}
  $('viewer').style.aspectRatio=`${width}/${height}`;
  setText('frameResolution',`${width} × ${height} px`);
  $('placeholder').hidden=true;
}
function distanceText(box){return Number.isFinite(box.distance)?`≈ ${box.distance.toFixed(2)} m`:box.distanceReason||'Set camera reference';}
function draw(result){
  const {width,height}=result;dimensions(width,height);overlayContext.clearRect(0,0,width,height);
  const displayScale=Math.max(.1,Math.min($('viewer').clientWidth/width,$('viewer').clientHeight/height));
  const font=Math.max(13,Math.round(width*.018),Math.ceil(11/displayScale));overlayContext.font=`500 ${font}px "Plex Mono", monospace`;
  overlayContext.lineWidth=Math.max(1.5,1.2/displayScale);
  for(const box of result.boxes){
    const color=COLORS[box.classId];overlayContext.strokeStyle=color;overlayContext.strokeRect(box.x1,box.y1,box.x2-box.x1,box.y2-box.y1);
    const text=Number.isFinite(box.distance)?`${distanceText(box)}`:CLASSES[box.classId];
    const tw=overlayContext.measureText(text).width+16,x=Math.max(0,Math.min(box.x1,width-tw)),y=Math.max(0,box.y1-font-13);
    overlayContext.fillStyle='#20231fe8';overlayContext.fillRect(x,y,tw,font+12);
    overlayContext.fillStyle=color;overlayContext.fillText(text,x+8,y+font+1);
    overlayContext.fillStyle='#e5e7dc';for(const [px,py] of box.posePoints||[]){overlayContext.beginPath();overlayContext.arc(px,py,Math.max(2,width*.002),0,Math.PI*2);overlayContext.fill();}
  }
  overlay.hidden=false;
}
function listDetections(result){
  const list=$('detections');list.replaceChildren();
  for(const box of result.boxes){
    const item=document.createElement('li');item.className=`detection-row ${box.classId===1?'red':'blue'}`;
    const name=document.createElement('span');name.className='detection-label';name.textContent=CLASSES[box.classId];
    const range=document.createElement('strong');range.className=Number.isFinite(box.distance)?'detection-distance':'detection-reason';range.textContent=distanceText(box);
    const confidence=document.createElement('span');confidence.className='detection-confidence';confidence.textContent=`${Math.round(box.score*100)}%`;confidence.title='Detector confidence';
    item.append(name,range,confidence);
    if(Number.isFinite(box.distance)){const detail=document.createElement('small');detail.className='detection-meta';detail.textContent=box.quality>=.65?'Clear edges':'Edges fitted';item.append(detail);}
    list.append(item);
  }
  setText('counts',result.boxes.length?`${result.boxes.filter(b=>b.classId===0).length} blue · ${result.boxes.filter(b=>b.classId===1).length} red`:'No cubes detected');
  $('emptyHint').hidden=result.boxes.length>0;
}
function collectReference(result,request,profile,width,height){
  if(referenceRequest!==request)return;
  if(performance.now()-request.startedAt>20000){referenceRequest=null;setText('calibrationState','No clear reference found. Show one whole cube with clear edges and try again.');controls();return;}
  if(!result.reference){setText('calibrationState',`${result.boxes[0]?.distanceReason||'Show exactly one whole cube'} · waiting for a clear frame`);return;}
  if(request.profile&&!sameCamera(request.profile,profile)){referenceRequest=null;setText('calibrationState','Camera settings changed. Set the reference again.');controls();return;}
  request.profile={...profile};request.samples.push(result.reference.focalNormalized*width/Math.max(width,height));
  setText('calibrationState',`Reference ${request.samples.length}/${SAMPLE_COUNT} · keep the measured distance`);
  if(request.samples.length>=SAMPLE_COUNT){
    try{
      reference=finishReference(request.samples,request.metres,request.profile);referenceRequest=null;revision++;
      try{localStorage.setItem(REFERENCE_KEY,JSON.stringify(reference));}catch{}
      referenceState();status('Reference saved. You can move and change viewing angle.');
    }catch(error){referenceRequest=null;setText('calibrationState',error.message);controls();}
  }
}
function commit(result,startedAt,number,snapshot,profile,isPhoto,request,matching){
  if(request)collectReference(result,request,profile,result.width,result.height);
  for(const box of result.boxes){
    box.distance=Number.isFinite(box.distance??box.metres)?(box.distance??box.metres):null;
    box.posePoints=box.posePoints??box.points;
    if(isPhoto||!matching||request){box.distance=null;box.distanceReason=isPhoto?'Use the calibrated live camera':reference&&!sameCamera(reference.profile,profile)?'Recalibrate camera':'Set camera reference';}
  }
  const now=performance.now();
  if(lastCommitAt&&live){const interval=now-lastCommitAt;intervals.push(interval);if(intervals.length>15)intervals.shift();$('cadence').dataset.intervalMs=String(interval);}
  lastCommitAt=now;
  lastSnapshot=snapshot;lastResult={...result,startedAt,frameNumber:number};stale=false;
  draw(lastResult);listDetections(lastResult);
  const total=performance.now()-startedAt;lastResult.totalMs=total;
  if(live)setText('freshness',`${Math.round(total)} ms old`);
  setText('latency',`${Math.round(total)} ms`);$('latency').dataset.totalMs=String(total);$('latency').dataset.distanceMs=String(result.poseMs??0);$('latency').dataset.frameNumber=String(number);
  $('latency').classList.toggle('slow',total>=1000);
  setText('cadence',live&&intervals.length?`${(1000/median(intervals)).toFixed(1)} /s`:'—');
  const hasRange=result.boxes.some(box=>Number.isFinite(box.distance));
  setText('qualityState',isPhoto?'Detection only':referenceRequest?'Setting reference':reference&&!sameCamera(reference.profile,profile)?'Recalibrate camera':hasRange?'Cube pose · live':reference?'Looking for clear edges':'Reference needed');
  if(!referenceRequest)status(isPhoto?'Photo processed. Use the live camera for calibrated distance.':live?'Live detection · move and change viewing angle freely.':'Camera paused.');
  history.push({frame:number,captureToDisplayMs:total,intervalMs:intervals.at(-1)??null,modelMs:result.modelMs,detectorMs:result.detectionMs,poseMs:result.poseMs,
    boxes:result.boxes.map(b=>({classId:b.classId,score:b.score,metres:b.distance,fitErrorPixels:b.fitErrorPixels,quality:b.quality,reason:b.distanceReason}))});
  if(history.length>240)history.shift();controls();
}
function schedule(){
  if(!live||inflight||document.hidden||video.readyState<2||videoSequence<=capturedSequence)return;
  processFrame();
}
function watchVideoFrames(){
  if(!live||callbackId!==null)return;
  const token=generation;
  const onFrame=()=>{
    callbackId=null;if(!live||token!==generation)return;
    if(video.requestVideoFrameCallback||video.currentTime!==lastMediaTime){lastMediaTime=video.currentTime;videoSequence++;schedule();}
    watchVideoFrames();
  };
  if(video.requestVideoFrameCallback)callbackId=video.requestVideoFrameCallback(onFrame);
  else callbackId=requestAnimationFrame(onFrame);
}
async function processFrame(){
  if(!live||inflight||video.readyState<2)return;
  inflight=true;capturedSequence=videoSequence;const token=generation,settings=revision,startedAt=performance.now(),number=++frameNumber;
  const {width,height}=frameSize(video.videoWidth,video.videoHeight),profile=cameraProfile(stream.getVideoTracks()[0].getSettings(),video.videoWidth,video.videoHeight);
  const request=referenceRequest,matching=Boolean(reference&&sameCamera(reference.profile,profile));
  let bitmap=null;
  try{
    if(!ready)await warmup();
    if(!live||token!==generation)return;
    bitmap=await createImageBitmap(video,{resizeWidth:width,resizeHeight:height,resizeQuality:'high'});
    if(!live||token!==generation||settings!==revision){bitmap.close();return;}
    // Keep an exact copy for Pause/Save. No pixel readback or neural work runs here.
    const snapshot=spareCanvas;if(snapshot.width!==width)snapshot.width=width;if(snapshot.height!==height)snapshot.height=height;
    snapshot.getContext('2d').drawImage(bitmap,0,0,width,height);
    const options={width,height,confidence:Number($('confidence').value)/100};
    if(request)options.referenceMetres=request.metres;
    else if(matching)options.focalNormalized=reference.focalNormalized*Math.max(width,height)/width;
    const result=await engine.infer(bitmap,options);bitmap=null;
    if(!live||token!==generation||settings!==revision)return;
    const previous=lastSnapshot;commit(result,startedAt,number,snapshot,profile,false,request,matching);spareCanvas=previous||document.createElement('canvas');
  }catch(error){
    if(live&&token===generation){
      // A poor outline during setup is recoverable on the next newest frame.
      if(request&&referenceRequest===request&&!/timed out|could not|runtime|worker/i.test(error.message)){
        if(performance.now()-request.startedAt>20000){referenceRequest=null;controls();}
        setText('calibrationState',`${error.message} Try a clearer view.`);
      }else{stopCamera();ready=false;setText('modelState','Retry needed');status(`Processing stopped: ${error.message}`,true);}
    }
  }finally{bitmap?.close();inflight=false;controls();schedule();}
}
async function startCamera(){
  if(live)return;
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia||!globalThis.OffscreenCanvas){status('Use a recent Chrome browser over HTTPS for camera processing.',true);return;}
  live=true;photoMode=false;const token=++generation;lastCommitAt=0;intervals.length=0;videoSequence=0;capturedSequence=0;lastMediaTime=-1;
  $('detections').replaceChildren();setText('freshness','Starting');setText('counts','Waiting for a frame');
  controls();status('Allow the rear camera to begin.');
  try{
    const camera=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30,max:30}}});
    if(!live||token!==generation){camera.getTracks().forEach(track=>track.stop());return;}
    stream=camera;
    camera.getVideoTracks()[0].addEventListener('ended',()=>{if(live&&token===generation){stopCamera();status('Camera disconnected. Start it again to resume.',true);}});
    video.srcObject=camera;video.hidden=false;photo.hidden=true;await video.play();
    if(!live||token!==generation)return;
    dimensions(video.videoWidth,video.videoHeight);overlayContext.clearRect(0,0,overlay.width,overlay.height);overlay.hidden=true;
    if(!ready)await warmup();
    if(live&&token===generation){status('Live detection · preparing the first frame.');navigator.wakeLock?.request('screen').then(lock=>{if(live)wakeLock=lock;else lock.release();}).catch(()=>{});videoSequence++;watchVideoFrames();schedule();}
  }catch(error){if(token===generation){stopCamera();const messages={NotAllowedError:'Camera permission denied. Allow it in Chrome site settings and try again.',NotFoundError:'No rear camera found. You can still choose a photo.',NotReadableError:'Camera is busy. Stop it in other tabs or apps, then try again.'};status(messages[error.name]||error.message,true);}}
}
function stopCamera(){
  live=false;generation++;revision++;
  if(callbackId!==null){if(video.cancelVideoFrameCallback)video.cancelVideoFrameCallback(callbackId);else cancelAnimationFrame(callbackId);callbackId=null;}
  stream?.getTracks().forEach(track=>track.stop());stream=null;video.srcObject=null;video.hidden=true;
  wakeLock?.release().catch(()=>{});wakeLock=null;
  if(referenceRequest){referenceRequest=null;referenceState();}
  if(lastSnapshot){photo.width=lastSnapshot.width;photo.height=lastSnapshot.height;photo.getContext('2d').drawImage(lastSnapshot,0,0);photo.hidden=false;}
  if(lastResult&&!stale)draw(lastResult);
  setText('freshness','Paused');controls();
}
async function processPhoto(){
  if(inflight||!photoMode)return;
  inflight=true;controls();const token=generation,settings=revision,startedAt=performance.now(),number=++frameNumber;let bitmap;
  try{
    if(!ready)await warmup();bitmap=await createImageBitmap(photo);
    const result=await engine.infer(bitmap,{width:photo.width,height:photo.height,confidence:Number($('confidence').value)/100});bitmap=null;
    if(token!==generation||settings!==revision)return;
    const snapshot=document.createElement('canvas');snapshot.width=photo.width;snapshot.height=photo.height;snapshot.getContext('2d').drawImage(photo,0,0);
    commit(result,startedAt,number,snapshot,null,true,null,false);setText('freshness','Photo');
  }catch(error){status(`Could not process this photo: ${error.message}`,true);}
  finally{bitmap?.close();inflight=false;controls();if(photoMode&&token===generation&&settings!==revision)queueMicrotask(processPhoto);}
}
$('start').addEventListener('click',startCamera);
$('stop').addEventListener('click',()=>{stopCamera();status('Camera paused. Restart whenever you are ready.');});
$('calibrate').addEventListener('click',()=>{
  revision++;if(referenceRequest){referenceRequest=null;referenceState();return;}
  const metres=Number($('referenceMetres').value);
  if(!Number.isFinite(metres)||metres<.3||metres>15){setText('calibrationState','Enter a measured distance between 0.3 and 15 metres.');return;}
  referenceRequest={metres,samples:[],profile:null,startedAt:performance.now()};
  setText('calibrationState','Show one whole cube at the measured distance. Collecting seven clear frames…');controls();
});
$('clearCalibration').addEventListener('click',()=>{
  revision++;referenceRequest=null;reference=null;try{localStorage.removeItem(REFERENCE_KEY);}catch{}
  for(const box of lastResult?.boxes||[]){box.distance=null;box.distanceReason='Set camera reference';}
  if(lastResult){draw(lastResult);listDetections(lastResult);}referenceState();
});
$('confidence').addEventListener('input',()=>setText('confidenceValue',`${$('confidence').value}%`));
$('confidence').addEventListener('change',()=>{revision++;if(photoMode&&!inflight)processPhoto();});
$('photoFile').addEventListener('change',async event=>{
  const file=event.target.files[0];event.target.value='';if(!file||inflight&&!live)return;
  stopCamera();photoMode=true;const token=generation;status('Opening photo…');
  try{
    const bitmap=await createImageBitmap(file);
    try{if(token!==generation)return;const {width,height}=frameSize(bitmap.width,bitmap.height);photo.width=width;photo.height=height;photo.getContext('2d').drawImage(bitmap,0,0,width,height);photo.hidden=false;dimensions(width,height);overlayContext.clearRect(0,0,width,height);}
    finally{bitmap.close();}
    // Any previous camera inference must finish before the photo enters the worker.
    while(inflight&&token===generation)await new Promise(resolve=>setTimeout(resolve,20));
    if(token===generation)await processPhoto();
  }catch(error){status(`Could not open this photo: ${error.message}`,true);controls();}
});
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('save').addEventListener('click',()=>{
  if(!lastSnapshot||!lastResult)return;
  const out=document.createElement('canvas');out.width=lastSnapshot.width;out.height=lastSnapshot.height;const ctx=out.getContext('2d');ctx.drawImage(lastSnapshot,0,0);
  if(!stale)ctx.drawImage(overlay,0,0);out.toBlob(blob=>{if(blob)download(blob,'cube-lens-pro.png');});
});
$('diagnostics').addEventListener('click',()=>{
  const report={version:'cube-lens-pro-1',createdAt:new Date().toISOString(),userAgent:navigator.userAgent,hardwareConcurrency:navigator.hardwareConcurrency,backend:backendInfo,
    camera:reference?{width:reference.profile.width,height:reference.profile.height,zoom:reference.profile.zoom,facingMode:reference.profile.facingMode}:null,
    reference:reference?{metres:reference.referenceMetres,focalNormalized:reference.focalNormalized,sampleSpread:reference.sampleSpread}:null,frames:history,
    timing:'Capture-to-display application processing, not network ping or sensor latency.',accuracy:'Approximate nearest-surface range for complete 35 cm cubes. Physical accuracy requires measured tests.'};
  download(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}),'cube-lens-pro-diagnostics.json');
});
function freshness(now){
  if(live&&lastResult&&now-lastPaintAt>100){
    lastPaintAt=now;const age=now-lastResult.startedAt;setText('freshness',`${Math.round(age)} ms old`);
    if(age>STALE_MS&&!stale){stale=true;overlayContext.clearRect(0,0,overlay.width,overlay.height);$('detections').replaceChildren();setText('qualityState','Waiting for a fresh frame');}
  }
  paintTick=requestAnimationFrame(freshness);
}
document.addEventListener('visibilitychange',()=>{if(document.hidden&&live){stopCamera();status('Camera paused while the page is in the background.');}});
window.addEventListener('pagehide',()=>{stopCamera();engine.dispose();cancelAnimationFrame(paintTick);});
referenceState();controls();requestAnimationFrame(freshness);warmup().catch(()=>{});
