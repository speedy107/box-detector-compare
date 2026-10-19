// Standard YOLO11 detection output: [1, 4 + classCount, anchors].
export const INPUT_SIZE = 320;
export const CLASSES = ['Blue box', 'Red box'];
export const COLORS = ['#5295ff', '#ff6c79'];

export function letterbox(width, height) {
  const scale = Math.min(INPUT_SIZE / width, INPUT_SIZE / height);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);
  return {scale, resizedWidth, resizedHeight,
    left: Math.floor((INPUT_SIZE - resizedWidth) / 2),
    top: Math.floor((INPUT_SIZE - resizedHeight) / 2), width, height};
}

export function rgbaToTensor(rgba) {
  const pixels = INPUT_SIZE * INPUT_SIZE;
  const data = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i++) {
    data[i] = rgba[4 * i] / 255;
    data[pixels + i] = rgba[4 * i + 1] / 255;
    data[2 * pixels + i] = rgba[4 * i + 2] / 255;
  }
  return data;
}

function iou(a, b) {
  const overlap = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)) *
    Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const union = (a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-overlap;
  return union > 0 ? overlap / union : 0;
}

export function decode(output, geometry, threshold = .35, nmsThreshold = .45) {
  if (output.dims.length !== 3 || output.dims[0] !== 1 || output.dims[1] !== 6) {
    throw new Error('Unexpected model output. Use the supplied blue/red box model.');
  }
  const count = output.dims[2], data = output.data, candidates = [];
  const clamp = (v, max) => Math.max(0, Math.min(max, v));
  for (let i=0; i<count; i++) {
    const blue = data[4*count+i], red = data[5*count+i];
    const classId = blue >= red ? 0 : 1, score = Math.max(blue,red);
    if (!Number.isFinite(score) || score < threshold) continue;
    const cx=data[i], cy=data[count+i], w=data[2*count+i], h=data[3*count+i];
    const x1=clamp((cx-w/2-geometry.left)/geometry.scale,geometry.width);
    const y1=clamp((cy-h/2-geometry.top)/geometry.scale,geometry.height);
    const x2=clamp((cx+w/2-geometry.left)/geometry.scale,geometry.width);
    const y2=clamp((cy+h/2-geometry.top)/geometry.scale,geometry.height);
    if (![x1,y1,x2,y2].every(Number.isFinite) || x2<=x1 || y2<=y1) continue;
    candidates.push({classId,score,x1,y1,x2,y2});
  }
  candidates.sort((a,b)=>b.score-a.score);
  const kept=[];
  for (const candidate of candidates.slice(0,300)) {
    if (!kept.some(box=>box.classId===candidate.classId && iou(box,candidate)>nmsThreshold)) {
      kept.push(candidate);
      if (kept.length>=30) break;
    }
  }
  return kept;
}
