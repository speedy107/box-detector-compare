import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import cvModule from '../vendor/opencv.js';
import '../pose-geometry.js';
import {restorePoseReference,samePoseCamera} from '../pose.js';

const cv=await cvModule;
const scenes=JSON.parse(await readFile(new URL('./fixtures/motion/scenes.json',import.meta.url),'utf8'));
for(const scene of scenes){
  const result=CubePoseGeometry.solve(cv,scene.points,scene.focal,640,480);
  assert.ok(result,`Pose must fit ${scene.name}`);
  assert.ok(Math.abs(result.metres/scene.truth_metres-1)<.001,`${scene.name}: camera rotation and range must be recovered`);
  assert.ok(result.error<.01);
}
const first=scenes[0],reference=CubePoseGeometry.calibrate(cv,first.points,first.truth_metres,640,480);
assert.ok(Math.abs(reference.focalNormalized-first.focal/640)<.001,'One measured reference recovers camera focal scale');
assert.equal(CubePoseGeometry.solve(cv,first.points.slice(0,5),500,640,480),null,'Ambiguous corner counts must not yield distances');
assert.equal(CubePoseGeometry.solve(cv,first.points,-1,640,480),null);
assert.throws(()=>CubePoseGeometry.calibrate(cv,first.points,0,640,480),/measured/);
const wrongSize=CubePoseGeometry.solve(cv,first.points,500,640,480,.7);
assert.ok(Math.abs(wrongSize.metres/first.truth_metres-2)<.001,'Metric scale depends on the real cube size');
const saved={version:1,focalBasis:'long-edge',sideMetres:.35,referenceMetres:3,focalNormalized:.8,profile:{deviceId:'back',width:640,height:480,zoom:1,facingMode:'environment'}};
assert.equal(restorePoseReference(saved),saved);
assert.equal(restorePoseReference({...saved,sideMetres:.5}),null);
assert.equal(restorePoseReference({...saved,focalNormalized:Infinity}),null);
assert.equal(restorePoseReference({...saved,referenceMetres:undefined}),null);
assert.equal(restorePoseReference({...saved,profile:{...saved.profile,deviceId:''}}),null);
assert.equal(samePoseCamera(saved.profile,{...saved.profile,width:480,height:640}),true,'Portrait/landscape transposition preserves the camera reference');
assert.equal(samePoseCamera(saved.profile,{...saved.profile,width:1280,height:720}),false,'A different capture format must not silently reuse intrinsics');
assert.equal(samePoseCamera(saved.profile,{...saved.profile,zoom:2}),false);
assert.equal(restorePoseReference({...saved,focalBasis:undefined}),null,'Old width-normalized references cannot survive orientation changes safely');
console.log('PASS: cube pose across yaw/pitch/roll and distances, one-time camera scale, invalid outlines, physical-size dependence and reference validation.');
