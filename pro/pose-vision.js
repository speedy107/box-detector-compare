/* Colour finds the silhouette; fitted boundary lines supply subpixel corners.
 * Geometry still has to explain the complete outline of a 35 cm cube. */
globalThis.CubePoseVision=(()=>{
  const clamp=(x,lo,hi)=>Math.max(lo,Math.min(hi,x));
  const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
  function polygonArea(points){let area=0;for(let i=0;i<points.length;i++){const a=points[i],b=points[(i+1)%points.length];area+=a[0]*b[1]-b[0]*a[1];}return Math.abs(area)/2;}
  function edgeDistance(point,a,b){
    const dx=b[0]-a[0],dy=b[1]-a[1],length2=dx*dx+dy*dy;
    const t=clamp(((point[0]-a[0])*dx+(point[1]-a[1])*dy)/Math.max(length2,1e-9),0,1);
    return Math.hypot(point[0]-a[0]-t*dx,point[1]-a[1]-t*dy);
  }
  function boundaryError(contour,polygon){
    let sum=0;for(const p of contour){let best=Infinity;for(let i=0;i<polygon.length;i++)best=Math.min(best,edgeDistance(p,polygon[i],polygon[(i+1)%polygon.length]));sum+=Math.min(best*best,100);}
    return Math.sqrt(sum/Math.max(1,contour.length));
  }
  function fitLine(points){
    if(points.length<4)return null;
    let selected=points,line=null;
    for(let iteration=0;iteration<3;iteration++){
      let x=0,y=0;for(const p of selected){x+=p[0];y+=p[1];}x/=selected.length;y/=selected.length;
      let xx=0,xy=0,yy=0;for(const p of selected){const dx=p[0]-x,dy=p[1]-y;xx+=dx*dx;xy+=dx*dy;yy+=dy*dy;}
      const angle=.5*Math.atan2(2*xy,xx-yy),nx=-Math.sin(angle),ny=Math.cos(angle);line={nx,ny,c:nx*x+ny*y};
      if(iteration===2)break;
      const residuals=points.map(p=>Math.abs(nx*p[0]+ny*p[1]-line.c)).sort((a,b)=>a-b),cutoff=Math.max(.7,2.5*residuals[Math.floor(residuals.length/2)]);
      const next=points.filter(p=>Math.abs(nx*p[0]+ny*p[1]-line.c)<=cutoff);if(next.length<4)break;selected=next;
    }
    return line;
  }
  function gradientBoundary(point,nx,ny,rgb,x,y){
    const px=point[0]-x,py=point[1]-y;
    const sample=(sx,sy,channel)=>{
      const ix=Math.floor(sx),iy=Math.floor(sy),dx=sx-ix,dy=sy-iy;
      if(ix<0||iy<0||ix+1>=rgb.cols||iy+1>=rgb.rows)return null;
      const stride=rgb.cols*3,offset=iy*stride+ix*3+channel,data=rgb.data;
      return data[offset]*(1-dx)*(1-dy)+data[offset+3]*dx*(1-dy)+data[offset+stride]*(1-dx)*dy+data[offset+stride+3]*dx*dy;
    };
    const scores=[];let best=0;
    for(let i=0;i<=12;i++){
      const shift=-3+i*.5;let score=0;
      for(let channel=0;channel<3;channel++){
        const a=sample(px+nx*(shift-.75),py+ny*(shift-.75),channel),b=sample(px+nx*(shift+.75),py+ny*(shift+.75),channel);
        if(a===null||b===null)return point;score+=(a-b)**2;
      }
      scores.push(score);if(score>scores[best])best=i;
    }
    if(scores[best]<36)return point;
    let shift=-3+best*.5;
    if(best>0&&best<12){const a=scores[best-1],b=scores[best],c=scores[best+1],denominator=a-2*b+c;if(Math.abs(denominator)>1e-6)shift+=clamp(.25*(a-c)/denominator,-.5,.5);}
    return [point[0]+nx*shift,point[1]+ny*shift];
  }
  function refinePolygon(polygon,contour,size,rgb,x,y){
    const lines=[],boundary=[],band=Math.max(1.5,size*.025),rgbImage={cols:rgb.cols,rows:rgb.rows,data:rgb.data};
    for(let i=0;i<polygon.length;i++){
      const a=polygon[i],b=polygon[(i+1)%polygon.length],dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy);
      if(length<4)return null;
      const selected=contour.filter(p=>{const t=((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(length*length);return t>=.12&&t<=.88&&Math.abs(dx*(p[1]-a[1])-dy*(p[0]-a[0]))/length<=band;});
      const stride=Math.max(1,Math.floor(selected.length/40)),edge=[];
      for(let j=0;j<selected.length;j+=stride)edge.push(gradientBoundary(selected[j],-dy/length,dx/length,rgbImage,x,y));
      const line=fitLine(edge);if(!line)return null;lines.push(line);boundary.push(...edge);
    }
    const refined=[];
    for(let i=0;i<polygon.length;i++){
      const a=lines[(i+lines.length-1)%lines.length],b=lines[i],det=a.nx*b.ny-a.ny*b.nx;
      if(Math.abs(det)<.12)return null;
      const p=[(a.c*b.ny-a.ny*b.c)/det,(a.nx*b.c-a.c*b.nx)/det];
      if(distance(p,polygon[i])>Math.max(3,size*.07))return null;refined.push(p);
    }
    // Reject an intersection that changes winding or folds a very short edge.
    let turn=0;for(let i=0;i<refined.length;i++){
      const a=refined[i],b=refined[(i+1)%refined.length],c=refined[(i+2)%refined.length],cross=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]);
      if(distance(a,b)<2||Math.abs(cross)<1e-5||(turn&&Math.sign(cross)!==turn))return null;turn=Math.sign(cross);
    }
    return {points:refined,boundary};
  }
  function saturationFloor(hsv,box,x,y,classId){
    const cols=hsv.cols,data=hsv.data,samples=[],left=Math.max(0,Math.floor(box.x1-x+(box.x2-box.x1)*.2)),top=Math.max(0,Math.floor(box.y1-y+(box.y2-box.y1)*.2));
    const right=Math.min(hsv.cols,Math.ceil(box.x2-x-(box.x2-box.x1)*.2)),bottom=Math.min(hsv.rows,Math.ceil(box.y2-y-(box.y2-box.y1)*.2));
    for(let py=top;py<bottom;py+=3)for(let px=left;px<right;px+=3){const offset=(py*cols+px)*3,h=data[offset],s=data[offset+1],v=data[offset+2];if(v>=25&&s>=30&&(classId===0?h>=85&&h<=145:h<=10||h>=170))samples.push(s);}
    if(samples.length<8)return 60;samples.sort((a,b)=>a-b);return clamp(Math.round(samples[Math.floor(samples.length/2)]*.45),32,60);
  }
  function corners(cv,image,box,width,height){
    const size=Math.min(box.x2-box.x1,box.y2-box.y1);
    if(box.score<.6||size<32)return {error:'Cube edges are too small or unclear'};
    if(box.x1<=2||box.y1<=2||box.x2>=width-2||box.y2>=height-2)return {error:'Show the whole cube'};
    const margin=Math.max(4,size*.1),x=Math.max(0,Math.floor(box.x1-margin)),y=Math.max(0,Math.floor(box.y1-margin));
    const right=Math.min(width,Math.ceil(box.x2+margin)),bottom=Math.min(height,Math.ceil(box.y2+margin));
    const roi=image.roi(new cv.Rect(x,y,right-x,bottom-y)),rgb=new cv.Mat(),hsv=new cv.Mat(),mask=new cv.Mat();
    const contours=new cv.MatVector(),hierarchy=new cv.Mat();let hull=null,contour=null;
    try{
      cv.cvtColor(roi,rgb,cv.COLOR_RGBA2RGB);cv.cvtColor(rgb,hsv,cv.COLOR_RGB2HSV);
      const saturation=saturationFloor(hsv,box,x,y,box.classId);
      const threshold=(lower,upper)=>{
        const low=new cv.Mat(hsv.rows,hsv.cols,hsv.type(),new cv.Scalar(...lower)),high=new cv.Mat(hsv.rows,hsv.cols,hsv.type(),new cv.Scalar(...upper)),out=new cv.Mat();
        try{cv.inRange(hsv,low,high,out);return out;}finally{low.delete();high.delete();}
      };
      if(box.classId===0){const m=threshold([85,saturation,25,0],[145,255,255,0]);m.copyTo(mask);m.delete();}
      else{const a=threshold([0,saturation,25,0],[10,255,255,0]),b=threshold([170,saturation,25,0],[179,255,255,0]);cv.bitwise_or(a,b,mask);a.delete();b.delete();}
      cv.findContours(mask,contours,hierarchy,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE);
      let largest=-1,area=0;
      for(let i=0;i<contours.size();i++){const c=contours.get(i),a=cv.contourArea(c);c.delete();if(a>area){area=a;largest=i;}}
      if(largest<0||area<(box.x2-box.x1)*(box.y2-box.y1)*.3)return {error:'Cube colour or corners unclear'};
      contour=contours.get(largest);hull=new cv.Mat();cv.convexHull(contour,hull);
      const contourData=contour.data32S,contourPoints=Array.from({length:contour.rows},(_,i)=>[contourData[2*i]+x,contourData[2*i+1]+y]);
      if(contourPoints.some(p=>p[0]<=x||p[0]>=right-1||p[1]<=y||p[1]>=bottom-1))return {error:'Cube outline touches another region or is clipped'};
      const solidity=area/Math.max(1,cv.contourArea(hull));
      if(solidity<.89)return {error:'Cube outline is hidden or broken'};
      const perimeter=cv.arcLength(hull,true),polygons=[],seen=new Set();
      for(const fraction of [.001,.002,.004,.006,.008,.01,.015,.02,.03]){
        const approx=new cv.Mat();cv.approxPolyDP(hull,approx,perimeter*fraction,true);
        const approximation=approx.data32S,points=Array.from({length:approx.rows},(_,i)=>[approximation[2*i]+x,approximation[2*i+1]+y]);approx.delete();
        if(![4,6].includes(points.length))continue;
        const key=points.flat().join(',');if(seen.has(key))continue;seen.add(key);
        const polygonFill=polygonArea(points)/Math.max(1,cv.contourArea(hull));
        if(polygonFill<.90)continue;
        polygons.push(points);
      }
      if(!polygons.length)return {error:'Cube corners unclear'};
      const candidates=[];
      // At most three distinct silhouettes; expensive pose solving is bounded.
      for(const points of polygons.slice(0,3)){
        const refined=refinePolygon(points,contourPoints,size,rgb,x,y),boundary=refined?.boundary||contourPoints;
        if(refined)candidates.push({points:refined.points,boundaryErrorPixels:boundaryError(boundary,refined.points),refined:true});
        candidates.push({points,boundaryErrorPixels:boundaryError(boundary,points),refined:false});
      }
      candidates.sort((a,b)=>a.boundaryErrorPixels-b.boundaryErrorPixels);
      return {points:candidates[0].points,candidates,size,solidity};
    }finally{roi.delete();rgb.delete();hsv.delete();mask.delete();contours.delete();hierarchy.delete();hull?.delete();contour?.delete();}
  }
  function reliable(result,size){return result&&result.error<=Math.max(1.5,size*.025)&&result.outside<=2.5;}
  function estimate(cv,detected,focal,width,height,referenceMetres){
    let best=null;
    for(const candidate of detected.candidates){
      let fitted;try{fitted=referenceMetres?CubePoseGeometry.calibrate(cv,candidate.points,referenceMetres,width,height,.35):CubePoseGeometry.solve(cv,candidate.points,focal,width,height,.35);}catch{continue;}
      if(!reliable(fitted,detected.size))continue;
      const score=fitted.error+candidate.boundaryErrorPixels*.8;
      if(!best||score<best.score)best={...fitted,...candidate,score};
    }
    return best;
  }
  function process(cv,rgba,width,height,boxes,options={}){
    const image=cv.matFromArray(height,width,cv.CV_8UC4,rgba),results=[];let reference=null;
    try{
      if(options.referenceMetres&&boxes.length!==1)throw new Error('Show exactly one cube while setting the camera reference.');
      for(const box of boxes){
        const detected=corners(cv,image,box,width,height);
        if(detected.error){results.push({metres:null,reason:detected.error});continue;}
        if(!options.referenceMetres&&!options.focalNormalized){results.push({metres:null,reason:'Set camera reference',points:detected.points});continue;}
        const fitted=estimate(cv,detected,options.focalNormalized*width,width,height,options.referenceMetres);
        if(!fitted){if(options.referenceMetres)throw new Error('The corners do not fit a 35 cm cube clearly enough. Try a clearer view.');results.push({metres:null,reason:'Cube corners do not fit reliably'});continue;}
        if(options.referenceMetres)reference={focalNormalized:fitted.focalNormalized,fitErrorPixels:fitted.error};
        const quality=clamp(1-(fitted.error+fitted.boundaryErrorPixels)/Math.max(2,detected.size*.05),0,1);
        results.push({metres:fitted.metres,reason:null,points:fitted.points,fitErrorPixels:fitted.error,boundaryErrorPixels:fitted.boundaryErrorPixels,minEdge:detected.size,quality});
      }
    }finally{image.delete();}
    return {boxes:results,reference};
  }
  return {corners,process};
})();
