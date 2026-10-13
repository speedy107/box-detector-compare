/* Depth runs in its own worker so the heavier model does not block camera controls. */
importScripts('./vendor/ort.wasm.min.js');
ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
ort.env.wasm.wasmPaths=new URL('./vendor/',self.location.href).href;
let sessionPromise=null;
const progress=message=>self.postMessage({type:'progress',message});

async function loadDepth(){
  if(!sessionPromise)sessionPromise=(async()=>{
    progress('Loading indoor depth model…');
    const metadataResponse=await fetch('./models/depth-anything-v2-metric-indoor-q8.json');
    if(!metadataResponse.ok)throw new Error(`Depth model information could not load (${metadataResponse.status}).`);
    const metadata=await metadataResponse.json();
    if(metadata.units!=='metres'||metadata.max_depth!==20||
       JSON.stringify(metadata.input_shape)!=='[1,3,518,518]'||
       JSON.stringify(metadata.output_shape)!=='[1,518,518]'||
       !/^[a-f0-9]{64}$/.test(metadata.onnx_sha256))throw new Error('Expected a verified indoor metric-depth model.');
    const response=await fetch('./models/depth-anything-v2-metric-indoor-q8.onnx');
    if(!response.ok)throw new Error(`Depth model download failed (${response.status}).`);
    const parts=[];let length=0,lastProgress=0;
    if(response.body){
      const reader=response.body.getReader();
      while(true){const {value,done}=await reader.read();if(done)break;parts.push(value);length+=value.length;
        if(performance.now()-lastProgress>200){lastProgress=performance.now();progress(`Loading indoor depth model · ${(length/1048576).toFixed(1)} MB…`);}}
    }else{const bytes=new Uint8Array(await response.arrayBuffer());parts.push(bytes);length=bytes.length;}
    const bytes=new Uint8Array(length);let offset=0;for(const part of parts){bytes.set(part,offset);offset+=part.length;}
    if(length!==metadata.onnx_bytes)throw new Error('Depth model download was incomplete. Retry depth to try again.');
    const digest=await crypto.subtle.digest('SHA-256',bytes);
    const sha=Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,'0')).join('');
    if(sha!==metadata.onnx_sha256)throw new Error('Depth model integrity check failed. Reload to download a fresh copy.');
    progress('Preparing indoor depth model…');
    return ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  })().catch(error=>{sessionPromise=null;throw error;});
  return sessionPromise;
}
self.onmessage=async({data})=>{
  const {id,tensor:values}=data;let tensor,outputs;
  try{
    const model=await loadDepth();tensor=new ort.Tensor('float32',values,[1,3,518,518]);
    const start=performance.now();outputs=await model.run({[model.inputNames[0]]:tensor});
    const elapsed=performance.now()-start,output=outputs[model.outputNames[0]];
    if(output.dims.length!==3||output.dims[0]!==1||output.dims[1]!==518||output.dims[2]!==518)throw new Error('Unexpected metric-depth map dimensions.');
    const depth=new Float32Array(output.data);
    if(!depth.every(value=>Number.isFinite(value)&&value>0&&value<=20))throw new Error('Depth model returned invalid metric values.');
    self.postMessage({id,result:{data:depth,width:518,height:518,elapsed}},[depth.buffer]);
  }catch(error){self.postMessage({id,error:error.message||String(error)});}
  finally{tensor?.dispose();if(outputs)Object.values(outputs).forEach(output=>output.dispose());}
};
