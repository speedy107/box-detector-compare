/* Pixel-level contour tests: independently projected known cube silhouettes.
 * These are synthetic image checks, not a measurement of phone accuracy. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import cvModule from '../../vendor/opencv.js';
import '../pose-geometry.js';
await import('../../pose-vision.js');
const previousVision=globalThis.CubePoseVision;
await import('../pose-vision.js');
const improvedVision=globalThis.CubePoseVision;
const cv=await cvModule;
const scenes=JSON.parse(await readFile(new URL('../../tests/fixtures/motion/scenes.json',import.meta.url),'utf8'));
const width=640,height=480;
function render(scene){
  const pixels=new Uint8Array(width*height*4),p=scene.points;
  for(let i=0;i<pixels.length;i+=4){pixels[i]=pixels[i+1]=pixels[i+2]=120;pixels[i+3]=255;}
  let direction=0;for(let i=0;i<p.length;i++){const a=p[i],b=p[(i+1)%p.length];direction+=a[0]*b[1]-b[0]*a[1];}direction=Math.sign(direction);
  const colour=scene.classId===0?[35,90,210]:[220,60,35];
  const [x1,y1,x2,y2]=scene.box;
  for(let y=Math.max(0,Math.floor(y1)-1);y<=Math.min(height-1,Math.ceil(y2)+1);y++)for(let x=Math.max(0,Math.floor(x1)-1);x<=Math.min(width-1,Math.ceil(x2)+1);x++){
    let inside=0;
    for(const dy of [.125,.375,.625,.875])for(const dx of [.125,.375,.625,.875]){
      if(p.every((a,i)=>{const b=p[(i+1)%p.length];return direction*((b[0]-a[0])*(y+dy-a[1])-(b[1]-a[1])*(x+dx-a[0]))>=0;}))inside++;
    }
    const amount=inside/16,index=(y*width+x)*4;for(let channel=0;channel<3;channel++)pixels[index+channel]=Math.round(120*(1-amount)+colour[channel]*amount);
  }
  return pixels;
}
function motionBlur(pixels,radius){
  const result=pixels.slice();
  for(let y=0;y<height;y++)for(let x=radius;x<width-radius;x++)for(let channel=0;channel<3;channel++){
    let sum=0;for(let dx=-radius;dx<=radius;dx++)sum+=pixels[(y*width+x+dx)*4+channel];result[(y*width+x)*4+channel]=Math.round(sum/(radius*2+1));
  }
  return result;
}
function desaturate(pixels,amount){
  const result=pixels.slice();for(let i=0;i<result.length;i+=4){const mean=(result[i]+result[i+1]+result[i+2])/3;for(let c=0;c<3;c++)result[i+c]=Math.round(mean+(result[i+c]-mean)*amount);}return result;
}
const box=scene=>({x1:scene.box[0],y1:scene.box[1],x2:scene.box[2],y2:scene.box[3],score:.95,classId:scene.classId});
const estimate=(vision,pixels,scene,options={focalNormalized:500/width})=>vision.process(cv,pixels,width,height,[box(scene)],options);
const images=scenes.map(render),reference=estimate(improvedVision,images[0],scenes[0],{referenceMetres:3}).reference;
assert.ok(Math.abs(reference.focalNormalized*width/500-1)<.03,'An image-based reference recovers focal scale within pixel precision');
const report={clean:[],motionBlur:[],desaturated:[]};
for(let i=0;i<scenes.length;i++){
  const scene=scenes[i],variants={clean:images[i],motionBlur:motionBlur(images[i],3),desaturated:desaturate(images[i],.3)};
  for(const [name,pixels] of Object.entries(variants)){
    const before=estimate(previousVision,pixels,scene).boxes[0],after=estimate(improvedVision,pixels,scene,{focalNormalized:reference.focalNormalized}).boxes[0];
    const relative=after.metres?Math.abs(after.metres/scene.truth_metres-1):null;
    report[name].push({scene:scene.name,previousMetres:before.metres,metres:after.metres,relativeError:relative});
    if(name==='clean')assert.ok(after.metres&&relative<.04,`${scene.name}: clean independently rendered cubes must keep accurate distances`);
    if(after.metres){assert.ok(relative<.08,`${scene.name} ${name}: an accepted image must not become an inaccurate distance`);assert.ok(Number.isFinite(after.quality)&&after.quality>=0&&after.quality<=1);assert.ok(Number.isFinite(after.boundaryErrorPixels));}
  }
}
const accepted=name=>report[name].filter(v=>v.metres).length;
assert.ok(accepted('desaturated')>=9,'A pale cube still has supported colour boundaries');
assert.ok(accepted('motionBlur')>=7,'Most moderate movement blur should still yield supported corners');
const lowScore={...box(scenes[0]),score:.2};
assert.equal(improvedVision.process(cv,images[0],width,height,[lowScore],{focalNormalized:reference.focalNormalized}).boxes[0].metres,null);
const clipped={...box(scenes[0]),x1:0};
assert.match(improvedVision.process(cv,images[0],width,height,[clipped],{focalNormalized:reference.focalNormalized}).boxes[0].reason,/whole cube/);
const wrongClass={...box(scenes[0]),classId:1};
assert.equal(improvedVision.process(cv,images[0],width,height,[wrongClass],{focalNormalized:reference.focalNormalized}).boxes[0].metres,null,'A wrong colour region must not yield a distance');
const hidden=images[0].slice(),s=scenes[0],cutX=Math.round((s.box[0]+s.box[2])/2),cutY=Math.round((s.box[1]+s.box[3])/2);
for(let y=cutY;y<cutY+30;y++)for(let x=cutX;x<cutX+30;x++){const i=(y*width+x)*4;hidden[i]=hidden[i+1]=hidden[i+2]=120;}
assert.equal(estimate(improvedVision,hidden,s).boxes[0].metres,null,'A large concave occlusion must not be hidden by the convex hull');
assert.throws(()=>improvedVision.process(cv,images[0],width,height,[box(s),box(s)],{referenceMetres:3}),/exactly one/);
console.log(JSON.stringify({accepted:{clean:accepted('clean'),motionBlur:accepted('motionBlur'),desaturated:accepted('desaturated')},observations:report},null,2));
