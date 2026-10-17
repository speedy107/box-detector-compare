importScripts('./pose-geometry.js?v=1','./pose-vision.js?v=1');
let ready=null;
async function runtime(){
  if(!ready)ready=(async()=>{
    self.postMessage({type:'progress',message:'Loading cube pose tools…'});
    importScripts('./vendor/opencv.js');
    return await self.cv;
  })().catch(error=>{ready=null;throw error;});
  return ready;
}
self.onmessage=async({data})=>{
  const {id,rgba,width,height,boxes,options}=data;
  try{
    const cv=await runtime(),start=performance.now();
    const result=CubePoseVision.process(cv,rgba,width,height,boxes,options);
    self.postMessage({id,result:{...result,elapsed:performance.now()-start}});
  }catch(error){self.postMessage({id,error:error.message||String(error)});}
};
