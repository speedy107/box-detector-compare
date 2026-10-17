export function restorePoseReference(value){
  return value?.version===1&&value.sideMetres===.35&&Number.isFinite(value.focalNormalized)&&value.focalNormalized>=.25&&value.focalNormalized<=5&&
    Number.isFinite(value.referenceMetres)&&value.referenceMetres>=.3&&value.referenceMetres<=15&&
    value.profile&&typeof value.profile.deviceId==='string'&&value.profile.deviceId.length>0&&
    Number.isFinite(value.profile.width)&&value.profile.width>0&&Number.isFinite(value.profile.height)&&value.profile.height>0&&
    Number.isFinite(value.profile.zoom)&&value.profile.zoom>0&&typeof value.profile.facingMode==='string'?value:null;
}
export function createPoseEstimator(onProgress=()=>{}){
  let worker=null,nextId=0;const pending=new Map();
  function reset(error){for(const p of pending.values()){clearTimeout(p.timeout);p.reject(error);}pending.clear();worker?.terminate();worker=null;}
  function ensure(){
    if(worker)return;
    worker=new Worker(new URL('./pose-worker.js?v=1',import.meta.url));
    worker.onmessage=({data})=>{
      if(data.type==='progress'){onProgress(data.message);return;}
      const p=pending.get(data.id);if(!p)return;pending.delete(data.id);clearTimeout(p.timeout);
      if(data.error)p.reject(new Error(data.error));else p.resolve(data.result);
    };
    worker.onerror=()=>reset(new Error('Cube pose tools could not start. Reload to retry.'));
    worker.onmessageerror=()=>reset(new Error('Cube pose results could not be read.'));
  }
  return {
    infer(rgba,width,height,boxes,options){
      ensure();const id=++nextId;
      return new Promise((resolve,reject)=>{
        const timeout=setTimeout(()=>reset(new Error('Cube pose processing timed out. Reload to retry.')),30000);
        pending.set(id,{resolve,reject,timeout});
        try{worker.postMessage({id,rgba,width,height,boxes,options},[rgba.buffer]);}catch(error){reset(error);}
      });
    },
    dispose(){reset(new Error('Cube pose processing stopped.'));},
  };
}
