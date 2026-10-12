import {INPUT_SIZE,CLASSES,COLORS,letterbox,rgbaToTensor,decode} from './detector.js';

const $=id=>document.getElementById(id);
const frame=document.createElement('canvas'),prep=document.createElement('canvas');
prep.width=prep.height=INPUT_SIZE;
const prepContext=prep.getContext('2d',{willReadFrequently:true});
const video=$('video');
let v5Promise=null,uploaded=null,stream=null,live=false,busy=false,hasFrame=false,epoch=0,rerun=false,frameNumber=0;
let lastResults=null;
const status=(message,error=false)=>{$('status').textContent=message;$('status').classList.toggle('error',error);};
function controls(){
  $('modelFile').disabled=busy; $('photoFile').disabled=busy;
  $('start').disabled=busy||live; $('stop').disabled=!live;
  $('save').disabled=busy||!lastResults?.uploaded;
}
function lock(value){busy=value;controls();}
function resetResults(){
  lastResults=null;$('save').disabled=true;
  for(const key of ['v5','uploaded']){$(`${key}Time`).textContent='—';$(`${key}Counts`).textContent=key==='uploaded'&&!uploaded?'Waiting for a model':'Ready to analyze';$(`${key}List`).replaceChildren();}
  $('agreement').textContent='Ready for the next comparison';
  $('agreementDetail').textContent='Both models will use the same captured frame.';
  for(const id of ['v5Canvas','uploadedCanvas'])if(hasFrame)$(id).getContext('2d').drawImage(frame,0,0);
}
async function makeSession(bytes){
  if(!globalThis.ort)throw new Error('The inference runtime could not load. Reload this page.');
  ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
  ort.env.wasm.wasmPaths=new URL('./vendor/',location.href).href;
  const session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  try{
    if(session.inputNames.length!==1||session.outputNames.length!==1)throw new Error('Expected one input and one detection output.');
    const tensor=new ort.Tensor('float32',new Float32Array(3*INPUT_SIZE*INPUT_SIZE).fill(114/255),[1,3,INPUT_SIZE,INPUT_SIZE]);
    let outputs;
    try{outputs=await session.run({[session.inputNames[0]]:tensor});const o=outputs[session.outputNames[0]];
      if(o.type!=='float32'||o.dims.length!==3||o.dims[0]!==1||o.dims[1]!==6||o.dims[2]<1)throw new Error('Expected a YOLO output shaped [1,6,N].');
    }finally{tensor.dispose();if(outputs)Object.values(outputs).forEach(o=>o.dispose());}
    return session;
  }catch(error){await session.release();throw error;}
}
async function loadV5(){
  if(!v5Promise)v5Promise=(async()=>{
    status('Loading the included V5 model…');
    const response=await fetch(new URL('./models/boxes-yolo11n-v5.onnx',location.href));
    if(!response.ok)throw new Error(`V5 download failed (${response.status}).`);
    return makeSession(await response.arrayBuffer());
  })().catch(error=>{v5Promise=null;throw error;});
  return v5Promise;
}
async function infer(session,tensor,geometry,confidence){
  let outputs;const start=performance.now();
  try{outputs=await session.run({[session.inputNames[0]]:tensor});const elapsed=performance.now()-start;
    return {boxes:decode(outputs[session.outputNames[0]],geometry,confidence,.45),elapsed};
  }finally{if(outputs)Object.values(outputs).forEach(o=>o.dispose());}
}
function capture(source,width,height){
  const scale=Math.min(1,1280/Math.max(width,height));
  frame.width=Math.max(1,Math.round(width*scale));frame.height=Math.max(1,Math.round(height*scale));
  frame.getContext('2d').drawImage(source,0,0,frame.width,frame.height);hasFrame=true;
}
function paintCanvas(canvas,result){
  canvas.width=frame.width;canvas.height=frame.height;canvas.hidden=false;
  canvas.parentElement.style.aspectRatio=`${frame.width}/${frame.height}`;
  canvas.parentElement.querySelector('.placeholder').hidden=true;
  const context=canvas.getContext('2d');context.drawImage(frame,0,0);
  if(!result)return;
  const font=Math.max(14,Math.round(frame.width*.025));context.font=`600 ${font}px system-ui`;context.lineWidth=Math.max(2,frame.width*.004);
  for(const box of result.boxes){
    context.strokeStyle=COLORS[box.classId];context.strokeRect(box.x1,box.y1,box.x2-box.x1,box.y2-box.y1);
    const text=`${CLASSES[box.classId]} ${Math.round(box.score*100)}%`,width=context.measureText(text).width+12;
    const x=Math.min(box.x1,Math.max(0,frame.width-width)),y=Math.max(0,box.y1-font-12);
    context.fillStyle=COLORS[box.classId];context.fillRect(x,y,width,font+12);context.fillStyle='#091321';context.fillText(text,x+6,y+font+2);
  }
}
function paintPanel(key,result){
  paintCanvas($(`${key}Canvas`),result);
  $(`${key}Time`).textContent=result?`${Math.round(result.elapsed)} ms`:'—';
  $(`${key}Counts`).textContent=result?(result.boxes.length?`${result.boxes.filter(b=>b.classId===0).length} blue · ${result.boxes.filter(b=>b.classId===1).length} red`:'No boxes detected'):'Choose an ONNX model to compare';
  const list=$(`${key}List`);list.replaceChildren();
  for(const box of result?.boxes||[]){const item=document.createElement('li');item.textContent=`${CLASSES[box.classId]} ${Math.round(box.score*100)}%`;item.classList.toggle('red',box.classId===1);list.append(item);}
}
function overlap(a,b){
  const intersection=Math.max(0,Math.min(a.x2,b.x2)-Math.max(a.x1,b.x1))*Math.max(0,Math.min(a.y2,b.y2)-Math.max(a.y1,b.y1));
  return intersection/((a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-intersection||1);
}
function agreement(a,b){
  if(!b){$('agreement').textContent='V5 result ready';$('agreementDetail').textContent='Choose an ONNX model to compare it on this photo.';return;}
  const edges=[];a.boxes.forEach((x,i)=>b.boxes.forEach((y,j)=>{const iou=overlap(x,y);if(x.classId===y.classId&&iou>=.5)edges.push({i,j,iou});}));
  edges.sort((x,y)=>y.iou-x.iou);const left=new Set(),right=new Set();
  for(const edge of edges)if(!left.has(edge.i)&&!right.has(edge.j)){left.add(edge.i);right.add(edge.j);}
  $('agreement').textContent=a.boxes.length+b.boxes.length?`${left.size} matched · ${a.boxes.length-left.size} only V5 · ${b.boxes.length-right.size} only your model`:'Both found no boxes';
  $('agreementDetail').textContent='Same captured frame · same confidence · matching class and at least 50% overlap. Check the image to judge which result is correct.';
}
async function compare(){
  if(busy||!hasFrame){rerun=true;return;}
  lock(true);rerun=false;const token=epoch,confidence=Number($('confidence').value)/100;let tensor;
  try{
    const baseline=await loadV5(),geometry=letterbox(frame.width,frame.height);
    prepContext.fillStyle='rgb(114,114,114)';prepContext.fillRect(0,0,INPUT_SIZE,INPUT_SIZE);
    prepContext.drawImage(frame,geometry.left,geometry.top,geometry.resizedWidth,geometry.resizedHeight);
    tensor=new ort.Tensor('float32',rgbaToTensor(prepContext.getImageData(0,0,INPUT_SIZE,INPUT_SIZE).data),[1,3,INPUT_SIZE,INPUT_SIZE]);
    // Alternate run order. The source tensor is shared and never recaptured between models.
    const result={v5:null,uploaded:null},order=frameNumber++%2?['uploaded','v5']:['v5','uploaded'];
    for(const key of order){const session=key==='v5'?baseline:uploaded;if(session)result[key]=await infer(session,tensor,geometry,confidence);}
    if(token!==epoch)return;
    lastResults=result;paintPanel('v5',result.v5);paintPanel('uploaded',result.uploaded);agreement(result.v5,result.uploaded);
    status(live?'Live comparison · both panels show the same captured frame':result.uploaded?'Comparison complete · files stayed on your device':'V5 analyzed your photo. Choose an ONNX file to compare.');
  }catch(error){if(token===epoch){stopCamera();status(`Could not compare: ${error.message}`,true);}}
  finally{tensor?.dispose();lock(false);if(rerun&&hasFrame&&!live){rerun=false;queueMicrotask(compare);}}
}
function stopCamera(message){
  live=false;epoch++;stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;controls();
  if(message)status(message);
}
async function cameraLoop(token){
  if(!live||token!==epoch)return;
  if(!busy&&video.readyState>=2){capture(video,video.videoWidth,video.videoHeight);await compare();}
  if(live&&token===epoch)setTimeout(()=>cameraLoop(token),140);
}
$('modelFile').addEventListener('change',async event=>{
  const file=event.target.files[0];event.target.value='';if(!file||busy)return;
  stopCamera();lock(true);status('Opening your model…');let replacement,changed=false;
  try{
    if(!file.name.toLowerCase().endsWith('.onnx'))throw new Error('Choose an .onnx file.');
    if(file.size>100*1024*1024)throw new Error('Choose a model smaller than 100 MB.');
    await loadV5();status('Checking your model…');replacement=await makeSession(await file.arrayBuffer());
    const previous=uploaded;uploaded=replacement;replacement=null;await previous?.release();
    $('modelName').textContent=`${file.name} · ${(file.size/1024/1024).toFixed(1)} MB · ready`;
    $('uploadedBadge').textContent='READY';resetResults();changed=true;status('Your model is ready. Choose a photo or start the camera.');
  }catch(error){await replacement?.release();status(`Could not load this model: ${error.message} Use a blue/red YOLO model with 320 × 320 input.`,true);}
  finally{lock(false);}
  if(hasFrame&&changed)await compare();
});
$('photoFile').addEventListener('change',async event=>{
  const file=event.target.files[0];event.target.value='';if(!file||busy)return;
  stopCamera();lock(true);status('Opening your photo…');
  try{const bitmap=await createImageBitmap(file);try{capture(bitmap,bitmap.width,bitmap.height);}finally{bitmap.close();}resetResults();}
  catch(error){status(`Could not open this photo: ${error.message}`,true);lock(false);return;}
  lock(false);await compare();
});
$('start').addEventListener('click',async()=>{
  if(live||busy)return;
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia){status('Camera access requires HTTPS and a browser with camera support. You can still choose a photo.',true);return;}
  live=true;const token=++epoch;lock(true);status('Allow camera access to compare live frames…');
  try{
    const camera=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}}});
    if(!live||token!==epoch){camera.getTracks().forEach(t=>t.stop());return;}
    stream=camera;video.srcObject=camera;await video.play();await loadV5();
    if(live&&token===epoch){lock(false);cameraLoop(token);}
  }catch(error){if(token===epoch){stopCamera();const messages={NotAllowedError:'Camera permission was denied. Allow it in browser settings or choose a photo.',NotFoundError:'No camera found. Choose a photo instead.',NotReadableError:'Camera is busy. Close other camera apps and try again.'};status(messages[error.name]||`Camera failed: ${error.message}`,true);}}
  finally{if(!live)lock(false);}
});
$('stop').addEventListener('click',()=>stopCamera('Camera stopped. Last comparison remains on screen.'));
$('confidence').addEventListener('input',()=>{$('confidenceValue').textContent=`${$('confidence').value}%`;});
$('confidence').addEventListener('change',()=>{if(hasFrame&&!live)compare();});
$('save').addEventListener('click',()=>{
  if(!lastResults?.uploaded||busy)return;
  const exportCanvas=document.createElement('canvas'),width=frame.width,height=frame.height;
  exportCanvas.width=width*2;exportCanvas.height=height+64;const c=exportCanvas.getContext('2d');
  c.fillStyle='#101827';c.fillRect(0,0,width*2,height+64);c.fillStyle='#edf2ff';c.font='22px system-ui';
  c.fillText(`V5 · ${Math.round(lastResults.v5.elapsed)} ms`,20,40);c.fillText(`Your model · ${Math.round(lastResults.uploaded.elapsed)} ms`,width+20,40);
  c.drawImage($('v5Canvas'),0,64);c.drawImage($('uploadedCanvas'),width,64);
  exportCanvas.toBlob(blob=>{if(!blob){status('Could not save the comparison.',true);return;}const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`box-comparison-${Date.now()}.png`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);},'image/png');
});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&live)stopCamera('Camera paused while this page was hidden. Start it again to continue.');});
window.addEventListener('pagehide',()=>stopCamera());
