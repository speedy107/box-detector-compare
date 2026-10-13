import assert from 'node:assert/strict';
import {depthTensor,medianBlockDepth,distanceLabel} from '../depth.js';

const tensor=depthTensor(new Uint8ClampedArray([255,0,0,255,0,255,0,255]));
assert.equal(tensor.length,6);
assert.ok(Math.abs(tensor[0]-(1-.485)/.229)<1e-6);
assert.ok(Math.abs(tensor[2]-(0-.456)/.224)<1e-6);
assert.ok(Math.abs(tensor[3]-(1-.456)/.224)<1e-6);

const map={width:10,height:10,data:new Float32Array(100).fill(15)};
for(let y=3;y<=7;y++)for(let x=3;x<=7;x++)map.data[y*10+x]=2;
map.data[55]=NaN;map.data[54]=Infinity;map.data[53]=0;
const box={x1:100,y1:100,x2:900,y2:900};
assert.equal(medianBlockDepth(box,map,1000,1000),2,'Central depth excludes surrounding distant background and invalid pixels');
assert.equal(medianBlockDepth({x1:10,y1:20,x2:90,y2:180},map,100,200),2,'Coordinate mapping supports different image aspect ratios');
assert.equal(medianBlockDepth({x1:2,y1:2,x2:2,y2:7},map,10,10),null);
assert.equal(medianBlockDepth(box,{...map,data:new Float32Array(100).fill(NaN)},1000,1000),null);
const sparse=new Float32Array(100).fill(NaN);sparse[44]=sparse[45]=sparse[46]=2;
assert.equal(medianBlockDepth(box,{...map,data:sparse},1000,1000),null,'A few valid pixels cannot support an otherwise invalid box');
assert.equal(medianBlockDepth(box,{...map,data:new Float32Array(100).fill(19.99)},1000,1000),null);
assert.equal(medianBlockDepth({x1:0,y1:0,x2:1,y2:1},map,1000,1000),null);
assert.equal(distanceLabel(2),'≈ 2.00 m');
assert.equal(distanceLabel(null),'Depth unavailable');
console.log('PASS: depth normalization, central median, coordinates, invalid/saturated depth, missing depth labels.');
