// Intrinsics use the longest frame edge so rotating the phone preserves scale.
export const REFERENCE_KEY='cube-lens-pro-reference-v1';
export const SAMPLE_COUNT=7;
export const median=values=>{const s=[...values].sort((a,b)=>a-b);return s[Math.floor(s.length/2)];};
export function cameraProfile(settings,width,height){
  return {deviceId:settings.deviceId||'',width:settings.width||width,height:settings.height||height,
    facingMode:settings.facingMode||'',zoom:Number.isFinite(settings.zoom)&&settings.zoom>0?settings.zoom:1};
}
export function sameCamera(a,b){
  return Boolean(a&&b&&a.deviceId===b.deviceId&&a.facingMode===b.facingMode&&Math.abs(a.zoom-b.zoom)<.001&&
    Math.min(a.width,a.height)===Math.min(b.width,b.height)&&Math.max(a.width,a.height)===Math.max(b.width,b.height));
}
export function restoreReference(v){
  return v?.version===1&&v.focalBasis==='long-edge'&&v.sideMetres===.35&&Number.isFinite(v.focalNormalized)&&v.focalNormalized>=.2&&v.focalNormalized<=5&&
    Number.isFinite(v.referenceMetres)&&v.referenceMetres>=.3&&v.referenceMetres<=15&&v.profile&&typeof v.profile.deviceId==='string'&&v.profile.deviceId.length>0&&
    Number.isFinite(v.profile.width)&&v.profile.width>0&&Number.isFinite(v.profile.height)&&v.profile.height>0&&
    Number.isFinite(v.profile.zoom)&&v.profile.zoom>0&&typeof v.profile.facingMode==='string'?v:null;
}
export function finishReference(samples,metres,profile){
  if(samples.length<SAMPLE_COUNT)throw new Error('Collect seven clear reference frames.');
  const focalNormalized=median(samples);
  if(samples.some(v=>!Number.isFinite(v)||Math.abs(v/focalNormalized-1)>.06))throw new Error('Reference varied too much. Hold at the measured distance and try again.');
  const reference=restoreReference({version:1,focalBasis:'long-edge',sideMetres:.35,focalNormalized,referenceMetres:metres,profile:{...profile},
    capturedAt:new Date().toISOString(),sampleSpread:Math.max(...samples)/Math.min(...samples)-1});
  if(!reference)throw new Error('Camera identity is unavailable. Try Chrome and allow camera access.');
  return reference;
}
