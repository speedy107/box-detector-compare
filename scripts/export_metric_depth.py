"""Export a verified official indoor Small checkpoint to browser-compatible ONNX.

Run with the Python environment used by box-detector-v5. The upstream source is
a build dependency outside this checkout. This script never changes that source.
"""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from PIL import Image

SOURCE_COMMIT = 'a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, type=Path)
parser.add_argument('--checkpoint', required=True, type=Path)
parser.add_argument('--sha256', required=True)
parser.add_argument('--checkpoint-revision', required=True)
parser.add_argument('--output', type=Path, default=Path('models/depth-anything-v2-metric-indoor-q8.onnx'))
parser.add_argument('--validation-image', action='append', type=Path, required=True)
args = parser.parse_args()
assert subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE_COMMIT
assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == args.sha256, 'Checkpoint integrity failed'
sys.path.insert(0, str(args.source / 'metric_depth'))
from depth_anything_v2.dpt import DepthAnythingV2

torch.set_num_threads(4)
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48,96,192,384], max_depth=20)
model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
model.eval()
args.output.parent.mkdir(parents=True, exist_ok=True)
float_output = args.output.with_name('depth-anything-v2-metric-indoor-fp32.onnx')
dummy = torch.zeros(1,3,518,518)
with torch.no_grad():
    torch.onnx.export(model, dummy, float_output, input_names=['images'], output_names=['depth'],
                      opset_version=17, dynamo=False, do_constant_folding=True)
onnx.checker.check_model(str(float_output))
quantize_dynamic(str(float_output), str(args.output), weight_type=QuantType.QInt8,
                 op_types_to_quantize=['MatMul'], per_channel=True,
                 extra_options={'MatMulConstBOnly': True})
onnx.checker.check_model(str(args.output))
options = ort.SessionOptions(); options.intra_op_num_threads = 4
fp = ort.InferenceSession(str(float_output), options, providers=['CPUExecutionProvider'])
q8 = ort.InferenceSession(str(args.output), options, providers=['CPUExecutionProvider'])
checks = []
for image_path in args.validation_image:
    image = np.asarray(Image.open(image_path).convert('RGB').resize((518,518), Image.Resampling.BICUBIC), dtype=np.float32)/255
    image = (image-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
    image = image.transpose(2,0,1)[None].copy()
    with torch.no_grad(): reference = model(torch.from_numpy(image)).numpy()
    start=time.perf_counter();float_depth=fp.run(None,{'images':image})[0];fp_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter();quant_depth=q8.run(None,{'images':image})[0];q8_ms=(time.perf_counter()-start)*1000
    assert reference.shape == float_depth.shape == quant_depth.shape == (1,518,518)
    assert np.isfinite(quant_depth).all() and (quant_depth>0).all() and (quant_depth<=20).all()
    max_float_error=float(np.max(np.abs(reference-float_depth)))
    relative_quant_error=float(np.mean(np.abs(float_depth-quant_depth)/np.maximum(float_depth,.1)))
    assert max_float_error < .01, (image_path, max_float_error)
    assert relative_quant_error < .05, (image_path, relative_quant_error)
    checks.append({'image':image_path.name,'image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'fp32_max_error_metres':max_float_error,
                   'q8_mean_relative_error':relative_quant_error,'fp32_ms':fp_ms,'q8_ms':q8_ms})
report={'name':'Depth Anything V2 Metric Hypersim Small','units':'metres','max_depth':20,
        'source_repository':'DepthAnything/Depth-Anything-V2','source_commit':SOURCE_COMMIT,
        'checkpoint_repository':'depth-anything/Depth-Anything-V2-Metric-Hypersim-Small',
        'checkpoint_revision':args.checkpoint_revision,'checkpoint_sha256':args.sha256,
        'input_shape':[1,3,518,518],'output_shape':[1,518,518],
        'preprocessing':'Whole-frame square resize to 518x518; RGB/255; ImageNet mean/std. Distinct from YOLO letterbox.',
        'quantization':'Per-channel signed int8 constant MatMul weights; other operations float32.',
        'onnx_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'onnx_bytes':args.output.stat().st_size,'validation':checks,
        'build_versions':{package:version(package) for package in ['torch','torchvision','onnx','onnxruntime','numpy','pillow','opencv-python-headless']},
        'validation_scope':'Numerical export and quantization comparisons. This does not validate physical distance accuracy.',
        'limitations':'Single-image learned optical-axis depth to the visible surface; not measured Euclidean range. Check against known distances with the target camera.'}
args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
