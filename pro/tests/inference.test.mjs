import assert from 'node:assert/strict';
import {after,test} from 'node:test';
import {createInferenceEngine} from '../inference.js';

// A controlled transport exercises cancellation and ownership races that are
// difficult to reproduce deterministically with a real, warmed browser worker.
const nativeWorker=globalThis.Worker;
class ControlledWorker {
  static instances=[];
  constructor(){this.requests=[];this.terminated=false;ControlledWorker.instances.push(this);}
  postMessage(message,transfer){
    if(this.failNext){this.failNext=false;throw new Error('Capture transfer failed');}
    this.requests.push({message,transfer});
  }
  reply(result,error){
    const {id}=this.requests.at(-1).message;
    this.onmessage({data:{id,result,error}});
  }
  terminate(){this.terminated=true;}
}
globalThis.Worker=ControlledWorker;
after(()=>{if(nativeWorker===undefined)delete globalThis.Worker;else globalThis.Worker=nativeWorker;});
const capture=()=>({closed:0,close(){this.closed++;}});
async function warmed(){
  const engine=createInferenceEngine(),ready=engine.warmup(),worker=ControlledWorker.instances.at(-1);
  worker.reply({ready:true});await ready;return {engine,worker};
}

test('simultaneous warmup requests share one worker initialization',async()=>{
  const engine=createInferenceEngine(),first=engine.warmup(),second=engine.warmup();
  const worker=ControlledWorker.instances.at(-1);
  assert.equal(first,second);assert.equal(worker.requests.length,1);
  worker.reply({ready:true});assert.deepEqual(await first,{ready:true});
  await engine.warmup();assert.equal(worker.requests.length,1);engine.dispose();
});

test('busy inference rejects another capture instead of building a queue',async()=>{
  const {engine,worker}=await warmed(),firstCapture=capture(),droppedCapture=capture();
  const first=engine.infer(firstCapture,{width:640,height:480});
  await assert.rejects(engine.infer(droppedCapture),/busy/);
  assert.equal(droppedCapture.closed,1);assert.equal(worker.requests.length,2);
  assert.equal(worker.requests.at(-1).transfer[0],firstCapture);
  worker.reply({boxes:[]});assert.deepEqual(await first,{boxes:[]});
  assert.equal(firstCapture.closed,1);engine.dispose();
});

test('dispose between warmup reply and continuation cannot submit a late capture',async()=>{
  const engine=createInferenceEngine(),bitmap=capture(),inference=engine.infer(bitmap);
  const worker=ControlledWorker.instances.at(-1),rejected=assert.rejects(inference,/stopped/);
  worker.reply({ready:true});engine.dispose();await rejected;
  assert.equal(worker.requests.length,1);assert.equal(bitmap.closed,1);assert.equal(worker.terminated,true);
});

test('startup failure permits retry and old worker errors cannot cancel the new worker',async()=>{
  const engine=createInferenceEngine(),first=engine.warmup(),oldWorker=ControlledWorker.instances.at(-1);
  const rejected=assert.rejects(first,/download failed/);
  oldWorker.reply(undefined,'Model download failed');await rejected;assert.equal(oldWorker.terminated,true);
  const retry=engine.warmup(),newWorker=ControlledWorker.instances.at(-1);
  assert.notEqual(newWorker,oldWorker);
  oldWorker.onerror({preventDefault(){}});
  assert.equal(newWorker.terminated,false);
  newWorker.reply({ready:true});assert.deepEqual(await retry,{ready:true});engine.dispose();
});

test('failed image transfer closes the untransferred capture and resets the worker',async()=>{
  const {engine,worker}=await warmed(),bitmap=capture();worker.failNext=true;
  await assert.rejects(engine.infer(bitmap),/transfer failed/);
  assert.equal(bitmap.closed,1);assert.equal(worker.terminated,true);engine.dispose();
});
