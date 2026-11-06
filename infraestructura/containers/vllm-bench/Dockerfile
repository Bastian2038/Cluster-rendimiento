FROM ipre/vllm:26.07-ray2.48.0
LABEL io.ipre.parent-image="sha256:da2b08e0cf7952aa6dc9c9f529abdb3577b641bcaeef54774be11f0fef321d6c"
COPY vllm-ray/verify_environment.py /opt/ipre/verify_environment.py
COPY vllm-bench/bench-requirements.txt /opt/ipre/bench-requirements.txt
RUN python3 /opt/ipre/verify_environment.py \
 && python3 -c "import numpy; assert numpy.__version__ == '2.1.0'" \
 && python3 -m pip install --no-deps --only-binary=:all: --no-cache-dir -r /opt/ipre/bench-requirements.txt \
 && python3 -m pip check \
 && python3 /opt/ipre/verify_environment.py \
 && python3 -c 'import pandas,numpy,io; assert pandas.__version__ == "2.3.3"; assert numpy.__version__ == "2.1.0"; rows=pandas.read_json(io.StringIO("{\"prompt\":\"CPU fixture\",\"output_tokens\":128}\n"),lines=True); assert rows.iloc[0]["output_tokens"] == 128'
