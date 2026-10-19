/* OpenCV is supplied by the caller. Works in a classic worker and Node tests. */
globalThis.CubePoseGeometry=(()=>{
  const unitCube=[];for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])unitCube.push([x,y,z]);
  const threeFaces=[[-1,-1,1],[-1,1,1],[-1,1,-1],[1,1,-1],[1,-1,-1],[1,-1,1]];
  const twoFaces=[[-1,-1,-1],[1,-1,-1],[1,-1,1],[1,1,1],[1,1,-1],[-1,1,-1]];
  const face=[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]];
  const scaled=(vertices,half)=>vertices.map(p=>p.map(v=>v*half));
  function project(vertices,rotation,translation,focal,width,height){
    return vertices.map(p=>{
      const q=[0,1,2].map(i=>rotation[i*3]*p[0]+rotation[i*3+1]*p[1]+rotation[i*3+2]*p[2]+translation[i]);
      return q[2]>0?[width/2+focal*q[0]/q[2],height/2+focal*q[1]/q[2]]:null;
    });
  }
  function outsideHull(points,polygon){
    let area=0;for(let i=0;i<polygon.length;i++){const p=polygon[i],q=polygon[(i+1)%polygon.length];area+=p[0]*q[1]-q[0]*p[1];}
    const sign=area>=0?1:-1;let outside=0;
    for(const point of points){
      if(!point)return Infinity;
      for(let i=0;i<polygon.length;i++){
        const a=polygon[i],b=polygon[(i+1)%polygon.length],dx=b[0]-a[0],dy=b[1]-a[1];
        outside=Math.max(outside,-sign*(dx*(point[1]-a[1])-dy*(point[0]-a[0]))/Math.max(1e-6,Math.hypot(dx,dy)));
      }
    }return outside;
  }
  function nearestSurface(rotation,centre,half){
    const camera=[0,1,2].map(j=>-(rotation[j]*centre[0]+rotation[3+j]*centre[1]+rotation[6+j]*centre[2]));
    return Math.hypot(...camera.map(v=>Math.max(Math.abs(v)-half,0)));
  }
  function solve(cv,points,focal,width,height,side=.35,checkHull=true){
    if(![4,6].includes(points.length)||!Number.isFinite(focal)||focal<=0)return null;
    const half=side/2,K=cv.matFromArray(3,3,cv.CV_64F,[focal,0,width/2,0,focal,height/2,0,0,1]);
    const distortion=cv.Mat.zeros(4,1,cv.CV_64F),rvec=new cv.Mat(),tvec=new cv.Mat(),matrix=new cv.Mat();
    let best=null;
    try{
      for(const unit of points.length===6?[threeFaces,twoFaces]:[face]){
        const object=scaled(unit,half),objectMat=cv.matFromArray(object.length,1,cv.CV_64FC3,object.flat());
        try{for(const reverse of [false,true])for(let offset=0;offset<points.length;offset++){
          const ordered=Array.from({length:points.length},(_,i)=>points[(offset+(reverse?-i:i)+points.length)%points.length]);
          const imageMat=cv.matFromArray(points.length,1,cv.CV_64FC2,ordered.flat());
          try{
            const ok=cv.solvePnP(objectMat,imageMat,K,distortion,rvec,tvec,false,points.length===6?cv.SOLVEPNP_SQPNP:cv.SOLVEPNP_ITERATIVE);
            if(!ok)continue;
            cv.Rodrigues(rvec,matrix);
            const rotation=Array.from(matrix.data64F),translation=Array.from(tvec.data64F);
            if(translation[2]<=half||!translation.every(Number.isFinite)||!rotation.every(Number.isFinite))continue;
            const projected=project(object,rotation,translation,focal,width,height);if(projected.some(p=>!p))continue;
            const error=Math.sqrt(projected.reduce((sum,p,i)=>sum+(p[0]-ordered[i][0])**2+(p[1]-ordered[i][1])**2,0)/points.length);
            const centre=[...translation];
            if(points.length===4){
              const direction=rotation[2]*translation[0]+rotation[5]*translation[1]+rotation[8]*translation[2]>0?1:-1;
              for(let i=0;i<3;i++)centre[i]+=rotation[i*3+2]*half*direction;
            }
            const vertices=project(scaled(unitCube,half),rotation,centre,focal,width,height),outside=outsideHull(vertices,points);
            if(checkHull&&outside>2.5)continue;
            const metres=nearestSurface(rotation,centre,half);
            if(!Number.isFinite(metres)||metres<=.05)continue;
            if(!best||error<best.error)best={metres,error,outside,focal,rotation,centre,vertices,model:points.length===4?'one face':unit===threeFaces?'three faces':'two faces'};
          }catch{}finally{imageMat.delete();}
        }}finally{objectMat.delete();}
      }
    }finally{K.delete();distortion.delete();rvec.delete();tvec.delete();matrix.delete();}
    return best;
  }
  function calibrate(cv,points,metres,width,height,side=.35){
    if(!Number.isFinite(metres)||metres<.3||metres>15)throw new Error('Enter a measured surface distance between 0.3 and 15 metres.');
    let low=width*.25,high=width*5,best=null;
    for(let i=0;i<16;i++){
      const focal=(low+high)/2,result=solve(cv,points,focal,width,height,side,false);
      if(!result){low=focal;continue;}
      const deviation=Math.abs(Math.log(result.metres/metres));
      if(!best||deviation<best.deviation)best={...result,deviation};
      if(result.metres<metres)low=focal;else high=focal;
    }
    if(!best||best.deviation>.025||best.outside>2.5)throw new Error('The cube corners do not support a reliable camera reference. Try a clearer view.');
    return {...best,focalNormalized:best.focal/width};
  }
  return {solve,calibrate,project,nearestSurface};
})();
