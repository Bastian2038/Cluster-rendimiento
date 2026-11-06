"""Functional CPU HTTP fixture; its timings are not model performance data."""
import json
from pathlib import Path
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

from transformers import AutoTokenizer
from infraestructura.inference_runtime import bench_command,model_path
from infraestructura.inference_metrics import summarize_bench_result

config=json.loads(Path('/code/experimentos/configs/inferencia-4-2.json').read_text())
tokenizers={m['key']:AutoTokenizer.from_pretrained(model_path(m),local_files_only=True,trust_remote_code=False)
            for m in config['models']}
counts={'generation_requests':0,'tokenize_requests':0}

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass

    def send_json(self,value):
        data=json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers();self.wfile.write(data)

    def do_GET(self):
        self.send_json({'object':'list','data':[{'id':key,'object':'model'} for key in tokenizers]})

    def do_POST(self):
        request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        model=request['model'];ids=tokenizers[model](request['prompt']).input_ids
        if self.path=='/tokenize':
            counts['tokenize_requests']+=1
            self.send_json({'tokens':ids,'count':len(ids),'max_model_len':4096});return
        assert self.path=='/v1/completions' and request['stream'] and request['max_tokens']==128
        counts['generation_requests']+=1
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        for index in range(128):
            event={'id':'cpu-fixture','object':'text_completion','created':1,'model':model,
                   'choices':[{'text':' a','index':0,'logprobs':None,'finish_reason':None}]}
            self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode());self.wfile.flush();time.sleep(.002)
        event={'id':'cpu-fixture','object':'text_completion','created':1,'model':model,'choices':[],
               'usage':{'prompt_tokens':len(ids),'completion_tokens':128,'total_tokens':len(ids)+128}}
        self.wfile.write(('data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n').encode());self.wfile.flush()

server=ThreadingHTTPServer(('127.0.0.1',18099),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
checks=[]
try:
    for model,length in zip(config['models'],(512,2048)):
        destination=Path('/work/http-fixture')/model['key']
        cmd=bench_command(config,model,length,1,'measured',str(destination))
        cmd[cmd.index('--base-url')+1]='http://127.0.0.1:18099'
        cmd[cmd.index('--num-prompts')+1]='2'
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'command.json').write_text(json.dumps({'command':cmd,'returncode':result.returncode,
            'stdout':result.stdout,'stderr':result.stderr},indent=2))
        if result.returncode:raise RuntimeError(result.stderr[-3000:])
        raw=json.loads((destination/'measured.json').read_text())
        metrics=summarize_bench_result(raw,expected_prompt_tokens=length,expected_count=2)
        checks.append({'model_key':model['key'],'input_tokens':length,'requests':2,
                       'output_tokens':128,'real_client_and_parser':'passed'})
    assert counts['generation_requests']==4,counts
finally:
    server.shutdown();server.server_close()
proof={'scope':'CPU loopback HTTP fixture; not GPU inference or performance evidence',
       'checks':checks,'counts':counts,'image_id':config['image']['id']}
with Path('/work/http-fixture-validation.json').open('x') as f:json.dump(proof,f,indent=2)
print(json.dumps(proof))
