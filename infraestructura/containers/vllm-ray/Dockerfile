# Build only after checking the parent image ID documented in README.md.
FROM ipre/vllm:26.07-pytorch-finalizer-c5f8ebc
LABEL io.ipre.parent.image-id="sha256:e30b26eee2e55ceadd9277910c923613e1cdf4197c1976ff9b33d28a3d214474" \
      io.ipre.ray.version="2.48.0" \
      io.ipre.tvm-ffi.version="0.1.9"
ENV VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
COPY verify_environment.py /opt/ipre-ray/verify_environment.py
COPY ray-requirements.txt /opt/ipre-ray/ray-requirements.txt
RUN python3 /opt/ipre-ray/verify_environment.py --before \
 && python3 -m pip install --no-deps --no-cache-dir --only-binary=:all: apache-tvm-ffi==0.1.9 \
 && python3 -m pip install --no-deps --no-cache-dir --only-binary=:all: --require-hashes -r /opt/ipre-ray/ray-requirements.txt \
 && python3 -m pip check \
 && python3 /opt/ipre-ray/verify_environment.py
