/* Color contours supply corners; cube geometry supplies pose, not box aspect. */
globalThis.CubePoseVision=(()=>{
  function corners(cv,image,box,width,height){
    const size=Math.min(box.x2-box.x1,box.y2-box.y1);
    if(box.score<.6||size<32)return {error:'Cube edges are too small or unclear'};
    if(box.x1<=2||box.y1<=2||box.x2>=width-2||box.y2>=height-2)return {error:'Show the whole cube'};
    const margin=Math.max(4,size*.1),x=Math.max(0,Math.floor(box.x1-margin)),y=Math.max(0,Math.floor(box.y1-margin));
    const right=Math.min(width,Math.ceil(box.x2+margin)),bottom=Math.min(height,Math.ceil(box.y2+margin));
    const roi=image.roi(new cv.Rect(x,y,right-x,bottom-y)),rgb=new cv.Mat(),hsv=new cv.Mat(),mask=new cv.Mat();
    const contours=new cv.MatVector(),hierarchy=new cv.Mat();let hull=null;
    try{
      cv.cvtColor(roi,rgb,cv.COLOR_RGBA2RGB);cv.cvtColor(rgb,hsv,cv.COLOR_RGB2HSV);
      const threshold=(lower,upper)=>{
        const low=new cv.Mat(hsv.rows,hsv.cols,hsv.type(),new cv.Scalar(...lower)),high=new cv.Mat(hsv.rows,hsv.cols,hsv.type(),new cv.Scalar(...upper)),out=new cv.Mat();
        try{cv.inRange(hsv,low,high,out);return out;}finally{low.delete();high.delete();}
      };
      if(box.classId===0){const m=threshold([85,60,25,0],[145,255,255,0]);m.copyTo(mask);m.delete();}
      else{const a=threshold([0,60,25,0],[10,255,255,0]),b=threshold([170,60,25,0],[179,255,255,0]);cv.bitwise_or(a,b,mask);a.delete();b.delete();}
      cv.findContours(mask,contours,hierarchy,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE);
      let largest=-1,area=0;
      for(let i=0;i<contours.size();i++){const c=contours.get(i),a=cv.contourArea(c);c.delete();if(a>area){area=a;largest=i;}}
      if(largest<0||area<(box.x2-box.x1)*(box.y2-box.y1)*.3)return {error:'Cube colour or corners unclear'};
      const contour=contours.get(largest);hull=new cv.Mat();cv.convexHull(contour,hull);contour.delete();
      const perimeter=cv.arcLength(hull,true);let four=null;
      for(const fraction of [.001,.002,.004,.006,.008,.01,.015,.02,.03]){
        const approx=new cv.Mat();cv.approxPolyDP(hull,approx,perimeter*fraction,true);
        const points=Array.from({length:approx.rows},(_,i)=>[approx.data32S[2*i]+x,approx.data32S[2*i+1]+y]);approx.delete();
        if(points.length===6)return {points,size};
        if(points.length===4&&!four)four={points,size};
      }
      return four||{error:'Cube corners unclear'};
    }finally{roi.delete();rgb.delete();hsv.delete();mask.delete();contours.delete();hierarchy.delete();hull?.delete();}
  }
  function reliable(result,size){return result&&result.error<=Math.max(1.5,size*.025)&&result.outside<=2.5;}
  function process(cv,rgba,width,height,boxes,options={}){
    const image=cv.matFromArray(height,width,cv.CV_8UC4,rgba),results=[];let reference=null;
    try{
      for(const box of boxes){
        const detected=corners(cv,image,box,width,height);
        if(detected.error){results.push({metres:null,reason:detected.error});continue;}
        const {points,size}=detected;
        if(options.referenceMetres){
          if(boxes.length!==1)throw new Error('Show exactly one cube while setting the camera reference.');
          const fitted=CubePoseGeometry.calibrate(cv,points,options.referenceMetres,width,height,.35);
          if(!reliable(fitted,size))throw new Error('The corners do not fit a 35 cm cube clearly enough. Try a clearer view.');
          reference={focalNormalized:fitted.focalNormalized,fitErrorPixels:fitted.error};
          results.push({metres:fitted.metres,reason:null,points});
        }else if(options.focalNormalized){
          const fitted=CubePoseGeometry.solve(cv,points,options.focalNormalized*width,width,height,.35);
          results.push(reliable(fitted,size)?{metres:fitted.metres,reason:null,points,fitErrorPixels:fitted.error}:{metres:null,reason:'Cube corners do not fit reliably'});
        }else results.push({metres:null,reason:'Set camera reference',points});
      }
    }finally{image.delete();}
    return {boxes:results,reference};
  }
  return {corners,process};
})();
