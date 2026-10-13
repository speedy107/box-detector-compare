import assert from 'node:assert/strict';
import {createDepthEstimator} from '../depth.js';

const timers=new Map(),workers=[];
let timerId=0;
const originalTimeout=globalThis.setTimeout,originalClear=globalThis.clearTimeout,originalWorker=globalThis.Worker;
globalThis.setTimeout=callback=>{const id=++timerId;timers.set(id,callback);return id;};
globalThis.clearTimeout=id=>timers.delete(id);
globalThis.Worker=class {
  constructor(){workers.push(this);}
  postMessage(message){this.message=message;}
  terminate(){this.terminated=true;}
};

try{
  const estimator=createDepthEstimator();
  const stalled=estimator.infer(new Float32Array(3));
  const timedOut=assert.rejects(stalled,/timed out/);
  [...timers.values()][0]();
  await timedOut;
  assert.equal(workers[0].terminated,true);
  assert.equal(timers.size,0);

  const retry=estimator.infer(new Float32Array(3));
  assert.equal(workers.length,2,'Retry uses a fresh worker after a timeout');
  workers[1].onmessage({data:{id:workers[1].message.id,result:{elapsed:1}}});
  assert.deepEqual(await retry,{elapsed:1});
  assert.equal(timers.size,0,'A completed inference clears its timeout');

  const stopped=estimator.infer(new Float32Array(3));
  const cancelled=assert.rejects(stopped,/stopped/);
  estimator.dispose();
  await cancelled;
  assert.equal(workers[1].terminated,true);
  assert.equal(timers.size,0);

  const unreadable=estimator.infer(new Float32Array(3));
  const messageError=assert.rejects(unreadable,/could not be read/);
  workers[2].onmessageerror();
  await messageError;
  assert.equal(workers[2].terminated,true);
  assert.equal(timers.size,0);
  console.log('PASS: stalled worker timeout, fresh retry, completion cleanup, cancellation, unreadable result recovery.');
}finally{
  globalThis.setTimeout=originalTimeout;globalThis.clearTimeout=originalClear;globalThis.Worker=originalWorker;
}
