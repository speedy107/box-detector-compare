/* Scoped, optional acceleration. Unsupported browsers continue with one thread. */
globalThis.cubeIsolationReady=(()=>{
  const scriptURL=document.currentScript?.src||location.href;
  const guardKey='cube-pro-isolation-reload-v1';
  const fallback=reason=>({isolated:false,threads:1,reason});
  const isolated=()=>({isolated:true,threads:Math.min(2,navigator.hardwareConcurrency||2)});
  if(globalThis.crossOriginIsolated){try{sessionStorage.removeItem(guardKey);}catch{}return Promise.resolve(isolated());}
  if(!globalThis.isSecureContext||!navigator.serviceWorker)return Promise.resolve(fallback('This browser uses single-thread processing.'));
  let attempted;
  try{attempted=sessionStorage.getItem(guardKey)==='1';}
  catch{return Promise.resolve(fallback('Private browser storage is unavailable.'));}
  if(attempted)return Promise.resolve(fallback('Optional camera acceleration is unavailable in this browser.'));
  const deadline=performance.now()+8000;
  function bounded(promise){
    return new Promise((resolve,reject)=>{
      const timeout=setTimeout(()=>reject(new Error('Camera acceleration setup timed out.')),Math.max(0,deadline-performance.now()));
      Promise.resolve(promise).then(value=>{clearTimeout(timeout);resolve(value);},error=>{clearTimeout(timeout);reject(error);});
    });
  }
  return (async()=>{
    try{
      const workerURL=new URL('./isolation-sw.js?v=1',scriptURL),scope=new URL('./',scriptURL);
      await bounded(navigator.serviceWorker.register(workerURL,{scope:scope.href,updateViaCache:'none'}));
      await bounded(new Promise((resolve,reject)=>{
        let timeout;
        const done=(error)=>{
          clearTimeout(timeout);navigator.serviceWorker.removeEventListener('controllerchange',changed);
          if(error)reject(error);else resolve();
        };
        const changed=()=>{
          const controller=navigator.serviceWorker.controller;
          if(!controller)return;
          const activeURL=new URL(controller.scriptURL);
          if(activeURL.origin===workerURL.origin&&activeURL.pathname===workerURL.pathname)done();
        };
        navigator.serviceWorker.addEventListener('controllerchange',changed);
        timeout=setTimeout(()=>done(new Error('Camera acceleration could not activate.')),Math.max(0,deadline-performance.now()));
        changed();
      }));
      if(globalThis.crossOriginIsolated)return isolated();
      // Persist before navigating. If storage is blocked, continuing is safer than
      // repeatedly reloading a page whose headers cannot be upgraded.
      sessionStorage.setItem(guardKey,'1');
      location.reload();
      // Normally navigation destroys this document. If it is prevented, unblock
      // the application instead of keeping the camera waiting indefinitely.
      return await new Promise(resolve=>setTimeout(()=>resolve(fallback('Continuing with single-thread processing.')),2000));
    }catch(error){return fallback(error.message||'Continuing with single-thread processing.');}
  })();
})();
