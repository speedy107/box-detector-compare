/* One captured image supplies both detection and cube pose. */
let runtimePromise=null,processing=false,frameCanvas=null,frameContext=null,prepCanvas=null,prepContext=null,inputData=null;

function progress(message){self.postMessage({type:'progress',message});}
function validateDimensions(bitmap,options){
  const width=options.width??bitmap.width,height=options.height??bitmap.height;
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width>4096||height>4096)
    throw new Error('The camera frame dimensions are invalid or too large.');
  return {width,height};
}
function runtime(){
  if(!runtimePromise)runtimePromise=(async()=>{
    if(typeof OffscreenCanvas!=='function')throw new Error('This browser needs OffscreenCanvas support. Update Chrome and try again.');
    progress('Loading the on-device detector…');
    // Begin the model download before parsing the computer vision runtime.
    const modelResponse=fetch(new URL('../models/best.onnx',self.location.href));
    // importScripts is available in this classic worker; detector.js remains the shared ES module.
    importScripts('../vendor/ort.wasm.min.js','./pose-geometry.js','./pose-vision.js');
    const detectorPromise=import('./detector.js');
    ort.env.wasm.numThreads=self.crossOriginIsolated?Math.min(2,navigator.hardwareConcurrency||2):1;
    ort.env.wasm.proxy=false;
    // Pthread workers must start inside the /pro/ service-worker scope to receive
    // isolation headers. Reuse the original binary; only its small loader is local.
    ort.env.wasm.wasmPaths={
      mjs:new URL('./vendor/ort-wasm-simd-threaded.mjs',self.location.href).href,
      wasm:new URL('../vendor/ort-wasm-simd-threaded.wasm',self.location.href).href,
    };
    const sessionPromise=(async()=>{
      const response=await modelResponse;
      if(!response.ok)throw new Error(`Detector download failed (${response.status}).`);
      return await ort.InferenceSession.create(await response.arrayBuffer(),{
        executionProviders:['wasm'],graphOptimizationLevel:'all',
      });
    })();
    progress('Preparing cube geometry and the detector…');
    const cvPromise=(async()=>{importScripts('../vendor/opencv.js');return await self.cv;})();
    // Wait for every initialization task so a rejected task cannot leak a live session.
    const [detectorResult,sessionResult,cvResult]=await Promise.allSettled([detectorPromise,sessionPromise,cvPromise]);
    const failed=[detectorResult,sessionResult,cvResult].find(result=>result.status==='rejected');
    if(failed){if(sessionResult.status==='fulfilled')await sessionResult.value.release();throw failed.reason;}
    const detector=detectorResult.value,session=sessionResult.value,cv=cvResult.value;
    let tensor,outputs;
    try{
      progress('Warming the detector…');
      const {INPUT_SIZE}=detector;
      prepCanvas=new OffscreenCanvas(INPUT_SIZE,INPUT_SIZE);
      prepContext=prepCanvas.getContext('2d',{willReadFrequently:true,alpha:false});
      frameCanvas=new OffscreenCanvas(1,1);
      frameContext=frameCanvas.getContext('2d',{willReadFrequently:true,alpha:false});
      if(!prepContext||!frameContext)throw new Error('This browser could not create the camera processing canvas.');
      inputData=new Float32Array(3*INPUT_SIZE*INPUT_SIZE).fill(114/255);
      tensor=new ort.Tensor('float32',inputData,[1,3,INPUT_SIZE,INPUT_SIZE]);
      outputs=await session.run({[session.inputNames[0]]:tensor});
      const output=outputs[session.outputNames[0]];
      if(output.dims.length!==3||output.dims[0]!==1||output.dims[1]!==6)
        throw new Error('The detector has an unsupported output format.');
      return {detector,session,cv};
    }catch(error){await session.release();throw error;}
    finally{tensor?.dispose();if(outputs)Object.values(outputs).forEach(output=>output.dispose());}
  })().catch(error=>{runtimePromise=null;throw error;});
  return runtimePromise;
}

async function infer(bitmap,options={}){
  let tensor,outputs;
  try{
    if(!bitmap||typeof bitmap.close!=='function')throw new Error('A captured camera image is required.');
    const {width,height}=validateDimensions(bitmap,options);
    const {detector,session,cv}=await runtime(),startedAt=performance.now();
    const {INPUT_SIZE,letterbox,decode}=detector;
    const confidence=options.confidence??.35;
    if(!Number.isFinite(confidence)||confidence<0||confidence>1)throw new Error('Detection confidence must be between zero and one.');
    if(options.referenceMetres!==undefined&&(!Number.isFinite(options.referenceMetres)||options.referenceMetres<.3||options.referenceMetres>15))
      throw new Error('Enter a measured reference distance between 0.3 and 15 metres.');
    if(options.focalNormalized!==undefined&&(!Number.isFinite(options.focalNormalized)||options.focalNormalized<=0))
      throw new Error('The saved camera reference is invalid. Set it again.');
    const geometry=letterbox(width,height),detectionStartedAt=performance.now();
    prepContext.fillStyle='rgb(114,114,114)';
    prepContext.fillRect(0,0,INPUT_SIZE,INPUT_SIZE);
    prepContext.drawImage(bitmap,geometry.left,geometry.top,geometry.resizedWidth,geometry.resizedHeight);
    const rgba=prepContext.getImageData(0,0,INPUT_SIZE,INPUT_SIZE).data,pixels=INPUT_SIZE*INPUT_SIZE;
    // Reuse the CHW buffer rather than generating 1.2 MB of garbage on every frame.
    for(let i=0;i<pixels;i++){
      inputData[i]=rgba[4*i]/255;
      inputData[pixels+i]=rgba[4*i+1]/255;
      inputData[2*pixels+i]=rgba[4*i+2]/255;
    }
    tensor=new ort.Tensor('float32',inputData,[1,3,INPUT_SIZE,INPUT_SIZE]);
    const modelStartedAt=performance.now();
    outputs=await session.run({[session.inputNames[0]]:tensor});
    const modelMs=performance.now()-modelStartedAt;
    const boxes=decode(outputs[session.outputNames[0]],geometry,confidence,.45);
    const detectionMs=performance.now()-detectionStartedAt;
    let reference=null,poseMs=0,poseBoxes=[];
    if(boxes.length&&(options.referenceMetres!==undefined||options.focalNormalized!==undefined)){
      const poseStartedAt=performance.now();
      if(frameCanvas.width!==width)frameCanvas.width=width;
      if(frameCanvas.height!==height)frameCanvas.height=height;
      frameContext.drawImage(bitmap,0,0,width,height);
      const pixels=frameContext.getImageData(0,0,width,height).data;
      const pose=CubePoseVision.process(cv,pixels,width,height,boxes,options);
      poseBoxes=pose.boxes;reference=pose.reference;poseMs=performance.now()-poseStartedAt;
    }
    return {
      width,height,reference,detectionMs,modelMs,poseMs,workerMs:performance.now()-startedAt,
      boxes:boxes.map((box,index)=>{
        const pose=poseBoxes[index],metres=pose?.metres??null;
        const reason=pose?.reason??(metres===null?'Set camera reference':null);
        return {...box,...pose,metres,distance:metres,reason,distanceReason:reason,points:pose?.points,posePoints:pose?.points};
      }),
    };
  }finally{
    tensor?.dispose();if(outputs)Object.values(outputs).forEach(output=>output.dispose());
    bitmap?.close?.();
  }
}

self.onmessage=async({data})=>{
  const {id,type,bitmap,options}=data;
  if(processing){bitmap?.close?.();self.postMessage({id,error:'Camera processing is already running. Wait for the current frame.'});return;}
  processing=true;
  try{
    let result;
    if(type==='warmup'){
      const {detector}=await runtime();
      result={ready:true,backend:'wasm',threads:ort.env.wasm.numThreads,inputSize:detector.INPUT_SIZE,offscreenCanvas:true,crossOriginIsolated:Boolean(self.crossOriginIsolated)};
    }else if(type==='infer')result=await infer(bitmap,options);
    else{bitmap?.close?.();throw new Error('Unknown camera processing request.');}
    self.postMessage({id,result});
  }catch(error){self.postMessage({id,error:error.message||String(error)});}
  finally{processing=false;}
};
