/* The client keeps one request in flight. The camera owns scheduling and freshness. */
export function createInferenceEngine(onProgress=()=>{}){
  let worker=null,pending=null,nextId=0,ready=false,readyInfo=null,warmupPromise=null,generation=0;
  function reset(error){
    const request=pending;pending=null;ready=false;readyInfo=null;warmupPromise=null;generation++;
    worker?.terminate();worker=null;
    if(request){clearTimeout(request.timeout);request.reject(error);}
  }
  function ensure(){
    if(worker)return;
    const activeWorker=worker=new Worker(new URL('./inference-worker.js?v=1',import.meta.url));
    activeWorker.onmessage=({data})=>{
      if(worker!==activeWorker)return;
      if(data.type==='progress'){onProgress(data.message);return;}
      const request=pending;if(!request||data.id!==request.id)return;
      pending=null;clearTimeout(request.timeout);
      if(data.error)request.reject(new Error(data.error));else request.resolve(data.result);
    };
    activeWorker.onerror=event=>{if(worker!==activeWorker)return;event.preventDefault?.();reset(new Error('The camera processing worker could not start. Update Chrome or retry.'));};
    activeWorker.onmessageerror=()=>{if(worker===activeWorker)reset(new Error('The camera processing results could not be read. Retry to restart the detector.'));};
  }
  function request(type,payload={},transfer=[]){
    if(pending)return Promise.reject(new Error('Camera processing is busy. Only the newest frame should be submitted.'));
    try{ensure();}catch(error){return Promise.reject(error);}
    const id=++nextId;
    return new Promise((resolve,reject)=>{
      const timeout=setTimeout(()=>reset(new Error('Camera processing timed out. Retry to restart the detector.')),30000);
      pending={id,type,resolve,reject,timeout};
      try{worker.postMessage({id,type,...payload},transfer);}catch(error){reset(error);}
    });
  }
  function warmup(){
    if(warmupPromise)return warmupPromise;
    if(ready)return Promise.resolve(readyInfo);
    const token=generation;
    const promise=request('warmup').then(info=>{
      if(token!==generation)throw new Error('Camera processing stopped.');
      ready=true;readyInfo=info;return info;
    }).catch(error=>{
      // A failed initialization gets a fresh worker on retry.
      if(warmupPromise===promise)reset(error);
      throw error;
    });
    warmupPromise=promise;
    return promise;
  }
  return {
    warmup,
    async infer(bitmap,options={}){
      // Taking ownership here guarantees that rejected or cancelled captures are closed too.
      const token=generation;
      try{
        if(pending&&pending.type!=='warmup')throw new Error('Camera processing is busy. Drop this frame and capture the next one.');
        if(!ready)await warmup();
        if(token!==generation)throw new Error('Camera processing stopped.');
        if(pending)throw new Error('Camera processing is busy. Drop this frame and capture the next one.');
        return await request('infer',{bitmap,options},[bitmap]);
      }finally{
        // Closing a successfully transferred (detached) ImageBitmap is harmless.
        // This also covers postMessage failures before ownership was transferred.
        bitmap?.close?.();
      }
    },
    dispose(){reset(new Error('Camera processing stopped.'));},
  };
}
