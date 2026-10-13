// Depth Anything V2 Metric Hypersim Small: RGB/ImageNet input, metres output.
export const DEPTH_SIZE=518;
export const MAX_DEPTH=20;
export const DEPTH_MODEL_URL='./models/depth-anything-v2-metric-indoor-q8.onnx';

export function depthTensor(rgba){
  const count=rgba.length/4,data=new Float32Array(count*3);
  const mean=[.485,.456,.406],std=[.229,.224,.225];
  for(let i=0;i<count;i++)for(let channel=0;channel<3;channel++)
    data[channel*count+i]=(rgba[i*4+channel]/255-mean[channel])/std[channel];
  return data;
}

export function medianBlockDepth(box,depth,imageWidth,imageHeight){
  const {data,width,height}=depth;
  if(!data||data.length!==width*height||imageWidth<=0||imageHeight<=0)return null;
  const boxWidth=box.x2-box.x1,boxHeight=box.y2-box.y1;
  if(boxWidth<=0||boxHeight<=0)return null;
  // Sample only the central 60% of the box. Image and depth share the same full-frame coordinates.
  const startX=Math.max(0,Math.ceil((box.x1+.2*boxWidth)/imageWidth*width));
  const endX=Math.min(width-1,Math.floor((box.x2-.2*boxWidth)/imageWidth*width));
  const startY=Math.max(0,Math.ceil((box.y1+.2*boxHeight)/imageHeight*height));
  const endY=Math.min(height-1,Math.floor((box.y2-.2*boxHeight)/imageHeight*height));
  const samples=[];
  for(let y=startY;y<=endY;y++)for(let x=startX;x<=endX;x++){
    const value=data[y*width+x];if(Number.isFinite(value)&&value>0&&value<MAX_DEPTH)samples.push(value);
  }
  const area=Math.max(0,endX-startX+1)*Math.max(0,endY-startY+1);
  if(samples.length<3||samples.length<area/2)return null;
  samples.sort((a,b)=>a-b);const middle=Math.floor(samples.length/2);
  const median=samples.length%2?samples[middle]:(samples[middle-1]+samples[middle])/2;
  // Near the model's training cap, a precise-looking numeric label would be misleading.
  return median>=MAX_DEPTH*.99?null:median;
}

export function distanceLabel(metres){
  return metres===null||!Number.isFinite(metres)?'Depth unavailable':`≈ ${metres.toFixed(2)} m`;
}

export function createDepthEstimator(onProgress=()=>{}){
  let worker=null,nextId=0;
  const pending=new Map();
  function reset(error){for(const request of pending.values()){clearTimeout(request.timeout);request.reject(error);}pending.clear();worker?.terminate();worker=null;}
  function ensureWorker(){
    if(worker)return;
    worker=new Worker(new URL('./depth-worker.js?v=metric2',import.meta.url));
    worker.onmessage=({data})=>{
      if(data.type==='progress'){onProgress(data.message);return;}
      const request=pending.get(data.id);if(!request)return;pending.delete(data.id);clearTimeout(request.timeout);
      if(data.error)request.reject(new Error(data.error));else request.resolve(data.result);
    };
    worker.onerror=()=>reset(new Error('Depth processing could not start on this browser. Try a desktop browser or turn off distance estimates.'));
    worker.onmessageerror=()=>reset(new Error('Depth results could not be read. Retry depth to try again.'));
  }
  return {
    infer(tensor){
      ensureWorker();const id=++nextId;
      return new Promise((resolve,reject)=>{
        const timeout=setTimeout(()=>reset(new Error('Depth processing timed out. Retry depth or turn off distance estimates.')),180000);
        pending.set(id,{resolve,reject,timeout});
        try{worker.postMessage({id,tensor},[tensor.buffer]);}catch(error){reset(error);}
      });
    },
    dispose(){reset(new Error('Depth processing stopped.'));},
  };
}
