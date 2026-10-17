// Pinhole scale geometry for equal-size blocks and an unchanged camera.
// The reference distance supplies metric scale; no neural depth inference is used.
export const REFERENCE_SAMPLES=5;
const finitePositive=value=>Number.isFinite(value)&&value>0;

export function cameraProfile(settings,width,height){
  return {deviceId:settings.deviceId||'',width:settings.width||width,height:settings.height||height,
    facingMode:settings.facingMode||'',zoom:finitePositive(settings.zoom)?settings.zoom:1};
}
export function sameCamera(a,b){
  return Boolean(a&&b&&a.deviceId===b.deviceId&&a.width===b.width&&a.height===b.height&&
    a.facingMode===b.facingMode&&Math.abs(a.zoom-b.zoom)<.001);
}
function validProfile(profile){
  return profile&&typeof profile.deviceId==='string'&&profile.deviceId.length>0&&
    typeof profile.facingMode==='string'&&finitePositive(profile.width)&&finitePositive(profile.height)&&finitePositive(profile.zoom);
}
export function referenceSample(box,width,height){
  if(!finitePositive(width)||!finitePositive(height)||!box||!Number.isFinite(box.score)||box.score<.6)
    return {error:'Use a clearly detected block'};
  const {x1,y1,x2,y2}=box,w=x2-x1,h=y2-y1;
  if(![x1,y1,x2,y2].every(Number.isFinite)||w<=0||h<=0)return {error:'Invalid block outline'};
  const margin=Math.max(2,Math.min(width,height)*.01);
  if(x1<=margin||y1<=margin||x2>=width-margin||y2>=height-margin)
    return {error:'Show the whole block'};
  if(Math.min(w,h)*320/Math.max(width,height)<16)return {error:'Block too small'};
  if(Math.abs((x1+x2)/2/width-.5)>.35||Math.abs((y1+y2)/2/height-.5)>.35)
    return {error:'Keep the block near the centre'};
  return {width:w/width,height:h/height};
}
const median=values=>{const sorted=[...values].sort((a,b)=>a-b);return sorted[Math.floor(sorted.length/2)];};
export function createCalibration(samples,metres,profile){
  if(!Number.isFinite(metres)||metres<.1||metres>30)throw new Error('Enter a measured distance between 0.1 and 30 metres.');
  if(!validProfile(profile))throw new Error('Camera identity is unavailable; this browser cannot safely reuse a reference.');
  if(samples.length!==REFERENCE_SAMPLES||samples.some(s=>!finitePositive(s.width)||!finitePositive(s.height)||s.width>=1||s.height>=1))
    throw new Error('Capture five complete block outlines.');
  const width=median(samples.map(s=>s.width)),height=median(samples.map(s=>s.height));
  if(samples.some(s=>Math.abs(s.width/width-1)>.06||Math.abs(s.height/height-1)>.06))
    throw new Error('The outline moved too much. Hold the camera and block steady, then retry.');
  return {version:1,metres,width,height,profile:{...profile}};
}
export function restoreCalibration(value){
  if(value?.version!==1||!finitePositive(value.metres)||value.metres<.1||value.metres>30||
    !finitePositive(value.width)||!finitePositive(value.height)||value.width>=1||value.height>=1||!validProfile(value.profile))return null;
  return {version:1,metres:value.metres,width:value.width,height:value.height,profile:{...value.profile}};
}
export function calibratedDistance(box,width,height,calibration,profile,boxes=[]){
  if(!calibration)return {metres:null,reason:'Calibrate first'};
  if(!profile)return {metres:null,reason:'Use the calibrated camera'};
  if(!sameCamera(calibration.profile,profile))return {metres:null,reason:'Recalibrate camera'};
  const sample=referenceSample(box,width,height);
  if(sample.error)return {metres:null,reason:sample.error};
  for(const other of boxes){
    if(other===box)continue;
    const overlap=Math.max(0,Math.min(box.x2,other.x2)-Math.max(box.x1,other.x1))*
      Math.max(0,Math.min(box.y2,other.y2)-Math.max(box.y1,other.y1));
    if(overlap/((box.x2-box.x1)*(box.y2-box.y1))>.1)return {metres:null,reason:'Separate overlapping blocks'};
  }
  const widthScale=calibration.width/sample.width,heightScale=calibration.height/sample.height;
  if(Math.abs(Math.log(widthScale/heightScale))>.18)return {metres:null,reason:'Match the reference angle'};
  // Equal projection scaling in both axes is required. Geometric mean reduces
  // sensitivity to a single noisy box edge; it cannot correct hidden edges/rotation.
  const metres=calibration.metres*Math.sqrt(widthScale*heightScale);
  return finitePositive(metres)?{metres,reason:null}:{metres:null,reason:'Distance unavailable'};
}
