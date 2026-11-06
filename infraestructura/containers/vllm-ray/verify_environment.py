"""CPU-only checks: preserve NVIDIA packages and the existing PyTorch fix."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform

parser=argparse.ArgumentParser()
parser.add_argument('--before',action='store_true')
args=parser.parse_args()
expected={
    'torch':'2.13.0a0+9186a08b2c.nv26.7.59513937',
    'vllm':'0.24.0+092c4842.nv26.7.59534043',
    'transformers':'5.6.1',
}
actual={name:metadata.version(name) for name in expected}
if actual!=expected or platform.machine()!='aarch64' or platform.python_version_tuple()[:2]!=('3','12'):
    raise RuntimeError('Parent image does not match the inspected ARM64/Python3.12 environment')
torch_source=metadata.distribution('torch').locate_file('torch/library.py')
if hashlib.sha256(torch_source.read_bytes()).hexdigest()!='7354f4efe2a1477fe18a98b555a6d2255b8b5e27310609582dcb2708a5f06a98':
    raise RuntimeError('PyTorch finalizer patch differs from the verified parent')
v2=metadata.distribution('vllm').locate_file('vllm/v1/executor/ray_executor_v2.py')
if not v2.is_file():raise RuntimeError('The NVIDIA fork has no Ray V2 executor')
if not args.before:
    import ray
    import vllm.envs
    if ray.__version__!='2.48.0' or not vllm.envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
        raise RuntimeError('Expected Ray version/V2 selector absent')
    if metadata.version('apache-tvm-ffi')!='0.1.9':
        raise RuntimeError('vLLM requires apache-tvm-ffi==0.1.9')
    actual['ray']=ray.__version__
    actual['apache-tvm-ffi']=metadata.version('apache-tvm-ffi')
print(json.dumps({'versions':actual,'phase':'before' if args.before else 'after',
                  'ray_v2_source_sha256':hashlib.sha256(v2.read_bytes()).hexdigest()}))
