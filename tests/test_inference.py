"""CPU fixtures for orchestration, fail-closed readiness and paired reports."""

import copy
import ast
import hashlib
import io
import json
import shlex
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from infraestructura import inference as inf
from infraestructura import inference_runtime as rt
from infraestructura import manifest
from infraestructura.inference_evidence import RUNNER_SOURCES
from infraestructura.inference_metrics import summarize_bench_result
from infraestructura.inference_plot import prepare_plot_data
from infraestructura.inference_workload import generate_workload, validate_vllm_reader, write_workload
from tests.test_inference_metrics import fixture as metrics_fixture
from tests.test_inference_workload import FakeTokenizer


CONFIG_4_2_1 = Path(__file__).resolve().parents[1] / 'experimentos/configs/inferencia-4-2-1.json'
MODEL_KEYS = ('qwen35-9b', 'llama31-8b')
ROSTERS = {
    'four': ['pegasus', 'octopus', 'griffin', 'draggon'],
    'pair': ['pegasus', 'octopus'],
    'single': ['pegasus'],
}


def command_result(stdout='', stderr='', returncode=0):
    return {'stdout':stdout,'stderr':stderr,'returncode':returncode,'timed_out':False}


def ready_probe(config, cluster_config, node):
    index=cluster_config['groups']['four'].index(node['node_id'])
    packages={'vllm':'0.24.0+nv','ray':'2.48.0','torch':'2.13.0','transformers':'5.6.1',
              'tokenizers':'0.22.2','huggingface-hub':'1.24.0','apache-tvm-ffi':'0.1.9','pandas':'2.3.3'}
    return {'node_id':node['node_id'],'hostname':node['expected_hostname'],'architecture':'aarch64',
        'uid':1002,'gid':1002,'groups':[1002,100],'user':'bastiangl',
        'image':command_result(json.dumps(config['image']['id'])+' "arm64"'),
        'models':{m['key']:{'complete':True,'metadata_sha256':{'config.json':'a'*64},
                  'shards':[{'name':'model.safetensors','size_bytes':100,'present':True,'blob_name':'b'*64}]}
                  for m in config['models']},
        'gpu':command_result(f'GPU-{index}, NVIDIA GB10, 580.173.02\n'),
        'gpu_apps':command_result(),'occupied_ports':[],'ray_pids':[],
        'memory_bytes':{'MemAvailable':100*1024**3},'rdma_devices':['uverbs0'],
        'network':command_result(json.dumps([{'addr_info':[{'local':config['network']['node_ips'][node['node_id']]}]}])),
        'container_probe':command_result(json.dumps({'python':'3.12.3','packages':packages}))}


def write_fixture_roster(directory, model, nodes, preflight):
    probes = {probe['node_id']: probe for probe in preflight['nodes']}
    participants = []
    model_path = rt.model_path(model)
    for rank, node_id in enumerate(nodes):
        probe = probes[node_id]
        person = {'node_id': node_id, 'rank': rank, 'hostname': probe['hostname'],
                  'gpu_uuid': probe['gpu']['stdout'].split(',')[0]}
        packages = json.loads(probe['container_probe']['stdout'])['packages']
        evidence = f'raw/participants/rank-{rank}.json'
        inf.write_new(directory / evidence, dict(
            person, stage='load_model', model_path=model_path, tokenizer_path=model_path,
            world_size=len(nodes), effective_tp=len(nodes), effective_pp=1, dtype='torch.bfloat16',
            versions={name: packages[name] for name in ('vllm', 'torch', 'ray')},
        ))
        participants.append(dict(person, evidence_path=evidence))
    manifest.register_participants(directory, participants)


class FixtureReader:
    """Read the synthetic JSONL in order, using the fixture's local tokenizer."""

    def __init__(self, *, dataset_path, **kwargs):
        self.rows = [json.loads(line) for line in Path(dataset_path).read_text().splitlines()]

    def sample(self, *, tokenizer, num_requests, **kwargs):
        return [SimpleNamespace(prompt=row['prompt'], prompt_len=len(tokenizer(row['prompt']).input_ids),
                                expected_output_len=row['output_tokens']) for row in self.rows[:num_requests]]


def write_fixture_profile(profile, workload, tokenizer, packages):
    write_workload(workload, profile)
    # Keep the 48 full token-ID arrays per profile compact on disk.
    metadata = profile / 'metadata.json'
    metadata.write_text(json.dumps(inf.read_json(metadata), separators=(',', ':')), encoding='utf-8')
    validate_vllm_reader(workload, profile, tokenizer, reader_cls=FixtureReader,
                         package_versions={k: packages[k] for k in ('vllm', 'pandas', 'transformers')})


def fixture_campaign_inputs(root, config, packages):
    prepared = root / 'prepared'
    alternate = root / 'alternate'
    for name in RUNNER_SOURCES:
        path = prepared / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'"""Fixture module for {name}."""\n', encoding='utf-8')
    generator_hash = manifest.sha256_file(prepared / 'infraestructura/inference_workload.py')
    for model in config['models']:
        tokenizer = FakeTokenizer()
        for length in config['workload']['input_tokens']:
            profile = Path('workload') / model['key'] / str(length)
            workload = generate_workload(
                tokenizer, model_id=model['id'], model_revision=model['revision'],
                tokenizer_id=model['tokenizer_id'], tokenizer_revision=model['tokenizer_revision'],
                input_tokens=length, seed=config['workload']['seed'],
            )
            workload.update(tokenizer_local_path=rt.model_path(model), generator_sha256=generator_hash,
                            chat_template_sha256=hashlib.sha256(b'<user>...<assistant>').hexdigest())
            write_fixture_profile(prepared / profile, workload, tokenizer, packages)
            if model['key'] == 'qwen35-9b':
                # Reorder two measured requests, then issue valid metadata and reader receipts for that order.
                rows = workload['rows']
                rows[-2], rows[-1] = rows[-1], rows[-2]
                write_fixture_profile(alternate / profile, workload, tokenizer, packages)

    def files(directory):
        return {p.relative_to(directory).as_posix(): p for p in directory.rglob('*') if p.is_file()}

    return files(prepared), files(alternate)


def write_fixture_cell(path, model, cell, measured_seconds, drift):
    length, concurrency = cell['input_tokens'], cell['concurrency']
    name = f'{length}-c{concurrency}'
    folder = path / 'raw/nodes/pegasus/metrics' / name
    folder.mkdir(parents=True)
    for phase, count, duration in (('warmup', 16, measured_seconds / 2), ('measured', 32, measured_seconds)):
        raw = metrics_fixture(input_tokens=length, count=count, duration=duration)
        raw.update(max_concurrency=concurrency, request_rate='inf', model_id=model['key'],
                   tokenizer_id=rt.model_path(model), endpoint_type='openai')
        if drift == 'client' and (name, phase) in (('512-c1', 'warmup'), ('512-c4', 'measured')):
            raw['max_concurrency'] = 8
        (folder / f'{phase}.json').write_text(json.dumps(raw, separators=(',', ':')), encoding='utf-8')
    summary = summarize_bench_result(raw, expected_prompt_tokens=length)
    if drift == 'client' and name == '2048-c8':
        summary['t32_seconds'] += 1
    return dict(cell, **summary, cell=name, status='valid')


class InferencePlanTests(unittest.TestCase):
    def setUp(self):
        self.config,self.cluster,self.cluster_path=inf.load_config()

    def test_plan_has_exact_order_counts_and_no_side_effects(self):
        with (mock.patch.object(rt,'execute',side_effect=AssertionError('process')),
              mock.patch.object(inf,'write_new',side_effect=AssertionError('write'))):
            plan=inf.make_plan()
        self.assertEqual(len(plan['jobs']),12)
        self.assertEqual([j['name'] for j in plan['jobs'][:4]],
            ['b1-qwen35-9b-four','b1-qwen35-9b-pair','b1-llama31-8b-four','b1-llama31-8b-pair'])
        self.assertEqual(sum(len(j['cells']) for j in plan['jobs']),72)
        self.assertEqual(plan['measured_requests'],2304)
        self.assertEqual(plan['maximum_occupation_seconds'],28800)
        for job in plan['jobs']:
            args=job['server_command']
            self.assertEqual(int(args[args.index('--tensor-parallel-size')+1]),len(job['nodes']))
            self.assertIn('--no-enable-prefix-caching',args)
            for cell in job['cells']:
                args=cell['commands']['measured']
                self.assertIn('--ignore-eos',args)
                self.assertIn('--skip-chat-template',args)
                self.assertIn('--save-detailed',args)

    def test_rejects_protocol_drift_and_floating_model(self):
        for change in ('revision','blocks','seed','network','precision'):
            config=copy.deepcopy(self.config)
            if change=='revision':config['models'][0]['revision']='main'
            if change=='blocks':config['measurement']['blocks']=1
            if change=='seed':config['workload']['seed']=0
            if change=='network':config['network']['environment']['NCCL_SOCKET_IFNAME']='eth0'
            if change=='precision':config['runtime']['dtype']='float16'
            with self.subTest(change=change),self.assertRaises(inf.InferenceError):
                inf.validate_config(config,self.cluster)

    def test_embedded_remote_scripts_compile_without_execution(self):
        for code in (rt.PROBE_SOURCE,inf.STAGE,inf.COLLECT,inf.WORKER_LOGS):
            compile(code,'embedded','exec')
        for node in ast.walk(ast.parse(rt.PROBE_SOURCE)):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in node.targets):
                compile(ast.literal_eval(node.value),'nested-container-probe','exec')

    def test_docker_pins_image_user_and_read_only_models(self):
        node=self.cluster['nodes'][0]
        probe=ready_probe(self.config,self.cluster,node)
        command=inf.container_command(node,self.config,probe,'/tmp/own','unique-own-name',
                                      ['ray','start'],seconds=50)
        self.assertIn('--pull=never',command)
        self.assertIn(self.config['image']['id'],command)
        self.assertIn('1002:1002',command)
        self.assertIn(self.config['paths']['hf_hub']+':/models:ro',command)
        self.assertIn('VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1',command)
        self.assertIn('FLASHINFER_WORKSPACE_BASE=/work/cache',command)
        self.assertNotIn('--privileged',command)
        self.assertEqual(command[-3:],['50s','ray','start'])

    def test_readiness_rejects_environment_and_snapshot_drift(self):
        records={n['node_id']:ready_probe(self.config,self.cluster,n) for n in self.cluster['nodes']}
        def remote(node,*args,**kwargs):return command_result(json.dumps(records[node['node_id']]))
        with mock.patch.object(rt,'remote',side_effect=remote):
            self.assertEqual(rt.preflight(self.cluster,self.config,container_probe=True)['status'],'ready_for_smoke')
            records['griffin']['models']['qwen35-9b']['metadata_sha256']['config.json']='c'*64
            result=rt.preflight(self.cluster,self.config,container_probe=True)
            self.assertTrue(any('identidades de snapshot' in e for e in result['errors']))
            packages=json.loads(records['octopus']['container_probe']['stdout'])
            packages['packages']['ray']=None
            records['octopus']['container_probe']=command_result(json.dumps(packages))
            result=rt.preflight(self.cluster,self.config,container_probe=True)
            self.assertTrue(any('Ray ausente' in e for e in result['errors']))

    def test_preflight_missing_dependency_never_enters_gpu_path(self):
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(rt,'preflight',return_value={'errors':['Ray ausente'],'nodes':[]}), \
             mock.patch.object(inf,'write_new'), \
             mock.patch.object(inf,'prepare_workloads',side_effect=AssertionError('prepare')), \
             mock.patch.object(inf,'run_job',side_effect=AssertionError('GPU')):
            with self.assertRaisesRegex(inf.InferenceError,'preflight bloqueado'):
                inf.run_campaign(results_root=temporary)

    def test_cleanup_refuses_foreign_label_and_unknown_state(self):
        for result in (command_result('a'*64+' other-label'),
                       command_result(stderr='permission denied',returncode=1)):
            with mock.patch.object(rt,'remote',return_value=result) as remote:
                with self.assertRaises(inf.InferenceError):
                    inf.cleanup_container(self.cluster['nodes'][0],'own',Path('/unused'))
                self.assertEqual(remote.call_count,1)

    def test_cleanup_requires_successful_stop_and_absence(self):
        with tempfile.TemporaryDirectory() as temporary:
            results=[command_result('a'*64+' own'),command_result(),
                     command_result(stderr='daemon unavailable',returncode=1),command_result()]
            with mock.patch.object(rt,'remote',side_effect=results),self.assertRaises(RuntimeError):
                inf.cleanup_container(self.cluster['nodes'][0],'own',Path(temporary))

    def test_deadline_stops_before_more_commands(self):
        with mock.patch.object(inf.time,'monotonic',return_value=20):
            with self.assertRaises(TimeoutError):inf.remaining(19,60)
            self.assertEqual(inf.remaining(25,60),5)

    def test_effective_budget_cannot_extend_approved_session(self):
        for budget in (28801,300,0,True):
            with self.subTest(budget=budget),mock.patch.object(rt,'preflight',side_effect=AssertionError('remote')):
                with self.assertRaisesRegex(inf.InferenceError,'límite efectivo'):
                    inf.run_campaign(session_limit_seconds=budget)

    def test_readiness_stops_on_dead_server_without_waiting_for_timeout(self):
        with mock.patch.object(rt,'remote',return_value=command_result('false 1')):
            with self.assertRaisesRegex(inf.InferenceError,'terminó antes'):
                inf.wait_ready(18000,'fixture',inf.time.monotonic()+60,
                    lambda:inf.ensure_running(self.cluster['nodes'][0],'own',inf.time.monotonic()+60))

    def test_telemetry_must_cover_measured_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); node=self.cluster['nodes'][0]
            path=root/'raw/nodes/pegasus/logs/telemetry.jsonl';path.parent.mkdir(parents=True)
            samples=[{'hostname':node['expected_hostname'],'timestamp_utc':f'2026-11-06T12:00:0{i}+00:00',
                      'monotonic_ns':i*10**9,'gpus':[{'uuid':'GPU-0'}],
                      'memory':{'MemAvailable_kib':100*1024**2}} for i in range(2)]
            path.write_text('\n'.join(json.dumps(s) for s in samples))
            cells=[{'cell':'512-c1','measurement_started_at':'2026-11-06T12:10:00+00:00',
                    'measurement_ended_at':'2026-11-06T12:11:00+00:00'}]
            errors=inf._telemetry_valid(root,[node],cells)
            self.assertTrue(any('no cubre' in e for e in errors))

    def test_run_cli_returns_nonzero_for_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);(path/'campaign.json').write_text('{"status":"partial"}')
            with mock.patch.object(inf,'run_campaign',return_value=path),mock.patch.object(inf.signal,'signal'):
                self.assertEqual(inf.main(['run','--execute']),2)

    def test_failed_staging_closes_failed_without_gpu_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); workloads=root/'workloads';workloads.mkdir()
            probes={'nodes':[ready_probe(self.config,self.cluster,n) for n in self.cluster['nodes']]}
            with mock.patch.object(manifest,'capture_git',return_value={'commit':'a'*40,'dirty':False}), \
                 mock.patch.object(manifest,'capture_nvidia',return_value={}), \
                 mock.patch.object(inf,'stage',side_effect=RuntimeError('fixture stage failed')), \
                 mock.patch.object(rt,'remote',side_effect=AssertionError('GPU command')):
                result=inf.run_job(self.config,self.cluster,inf.DEFAULT_CONFIG,self.cluster_path,
                    next(inf.jobs(self.config)),'fixture-campaign',probes,workloads,
                    inf.time.monotonic()+60,root/'raw')
            self.assertEqual(result['status'],'failed')
            directory=Path(result['run_dir'])
            self.assertTrue((directory/'FAILED').is_file())
            self.assertFalse((directory/'COMPLETE').exists())
            manifest.verify_artifacts(directory)

    def test_interrupt_during_launch_cleans_registered_intent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);workloads=root/'workloads';workloads.mkdir()
            probes={'nodes':[ready_probe(self.config,self.cluster,n) for n in self.cluster['nodes']]}
            with mock.patch.object(manifest,'capture_git',return_value={'commit':'a'*40,'dirty':False}), \
                 mock.patch.object(manifest,'capture_nvidia',return_value={}), \
                 mock.patch.object(inf,'stage'), \
                 mock.patch.object(rt,'remote',side_effect=KeyboardInterrupt('fixture')), \
                 mock.patch.object(inf,'cleanup_container',return_value={'status':'confirmed_absent'}) as cleanup, \
                 mock.patch.object(inf,'collect'):
                result=inf.run_job(self.config,self.cluster,inf.DEFAULT_CONFIG,self.cluster_path,
                    next(inf.jobs(self.config)),'fixture-campaign',probes,workloads,
                    inf.time.monotonic()+60,root/'raw')
            self.assertEqual(result['status'],'failed')
            self.assertEqual(cleanup.call_count,1)
            self.assertEqual(cleanup.call_args.args[0]['node_id'],'pegasus')

    def test_prepare_interrupt_records_intent_closure_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            config=copy.deepcopy(self.config)
            config['paths']['workspace']=temporary
            probes={'nodes':[ready_probe(config,self.cluster,n) for n in self.cluster['nodes']]}
            with mock.patch.object(inf,'stage'),mock.patch.object(inf,'source_files',return_value={}), \
                 mock.patch.object(rt,'remote',side_effect=KeyboardInterrupt('fixture')), \
                 mock.patch.object(inf,'cleanup_container',return_value={'status':'confirmed_absent'}) as cleanup:
                with self.assertRaises(KeyboardInterrupt):
                    inf.prepare_workloads(config,self.cluster,probes,'fixture-campaign')
            root=Path(temporary)/'fixture-campaign'
            intent=json.loads((root/'preparation-intent.json').read_text())
            closure=json.loads((root/'preparation-closure.json').read_text())
            self.assertEqual(len(intent['commands']),4)
            self.assertFalse(intent['gpu'])
            self.assertEqual(intent['watchdog_seconds'],100)
            self.assertEqual(closure['status'],'failed')
            self.assertIn('KeyboardInterrupt',closure['errors'][0])
            self.assertEqual(cleanup.call_count,1)

    def test_input_copy_failure_closes_failed_before_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);workloads=root/'workloads';workloads.mkdir()
            probes={'nodes':[ready_probe(self.config,self.cluster,n) for n in self.cluster['nodes']]}
            with mock.patch.object(manifest,'capture_git',return_value={'commit':'a'*40,'dirty':False}), \
                 mock.patch.object(manifest,'capture_nvidia',return_value={}), \
                 mock.patch.object(Path,'read_bytes',side_effect=OSError('fixture copy failed')), \
                 mock.patch.object(inf,'stage',side_effect=AssertionError('GPU')):
                result=inf.run_job(self.config,self.cluster,inf.DEFAULT_CONFIG,self.cluster_path,
                    next(inf.jobs(self.config)),'fixture-campaign',probes,workloads,
                    inf.time.monotonic()+60,root/'raw')
            self.assertEqual(result['status'],'failed')
            self.assertTrue((Path(result['run_dir'])/'FAILED').is_file())
            manifest.verify_artifacts(result['run_dir'])

    def test_report_only_pairs_verified_equal_work_and_detects_raw_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); workload=root/'dataset';workload.write_text('fixture')
            refs=[]; preflight={'nodes':[ready_probe(self.config,self.cluster,n) for n in self.cluster['nodes']]}
            for job in list(inf.jobs(self.config))[:2]:
                model=job['model']; nodes=self.cluster['groups'][job['group']]
                with mock.patch.object(manifest,'capture_git',return_value={'commit':'a'*40,'dirty':False}), \
                     mock.patch.object(manifest,'capture_system',return_value={}), \
                     mock.patch.object(manifest,'capture_nvidia',return_value={}):
                    directory=manifest.start_run(kind='inference-serving',model_id=model['id'],
                        model_revision=model['revision'],tokenizer_id=model['id'],tokenizer_revision=model['revision'],
                        requested_config={'job':job,'protocol':self.config,'campaign_id':'fixture'},
                        inputs={'workload/fixture':workload},results_root=root/'raw')
                write_fixture_roster(directory, model, nodes, preflight)
                raw=metrics_fixture()
                if job['group']=='pair':
                    raw.update(duration=8.,request_throughput=4.,output_throughput=512.)
                cell=dict(summarize_bench_result(raw,expected_prompt_tokens=512),
                          input_tokens=512,concurrency=1,status='valid')
                inf.write_new(directory/'metrics/assessment.json',{'valid':True,'cells':[cell]})
                inf.write_new(directory/'raw/preflight.json',preflight)
                manifest.finalize_run(directory)
                refs.append({'job':job['name'],'group':job['group'],'block':job['block'],
                             'model_key':model['key'],'run_dir':str(directory),'status':'complete',
                             'manifest_sha256':manifest.sha256_file(directory/'manifest.json')})
            campaign=root/'campaign';campaign.mkdir()
            inf.write_new(campaign/'campaign.json',{'campaign_id':'fixture','status':'partial',
                          'pending_jobs':['pending'],'runs':refs,'protocol':self.config})
            inf.write_new(campaign/'plan.json',{})
            inf.write_new(campaign/'preflight.json',preflight)
            inf.write_new(campaign/'PARTIAL',{k:manifest.sha256_file(campaign/p) for k,p in
                [('campaign_sha256','campaign.json'),('plan_sha256','plan.json'),('preflight_sha256','preflight.json')]})
            report=inf.build_report(campaign)
            self.assertNotIn('excluded_cells', report)
            self.assertEqual(report['paired_comparisons'], [
                {'model': 'qwen35-9b', 'block': 1, 'input_tokens': 512, 'concurrency': 1,
                 's4_2': .5, 'e4_2': .25},
            ])
            def update_campaign(changed):
                (campaign/'campaign.json').write_text(json.dumps(changed))
                inf.write_new(campaign/'marker-next.json',{k:manifest.sha256_file(campaign/p) for k,p in
                    [('campaign_sha256','campaign.json'),('plan_sha256','plan.json'),('preflight_sha256','preflight.json')]})
                (campaign/'PARTIAL').write_bytes((campaign/'marker-next.json').read_bytes())
                (campaign/'marker-next.json').unlink()
            original=json.loads((campaign/'campaign.json').read_text())
            changed=copy.deepcopy(original);changed['campaign_id']='foreign'
            update_campaign(changed)
            with self.assertRaisesRegex(inf.InferenceError,'otra campaña'):inf.build_report(campaign)
            changed=copy.deepcopy(original);changed['runs'].append(copy.deepcopy(refs[0]))
            update_campaign(changed)
            with self.assertRaisesRegex(inf.InferenceError,'duplicada'):inf.build_report(campaign)
            update_campaign(original)
            (Path(refs[0]['run_dir'])/'metrics/assessment.json').write_text('{}')
            with self.assertRaises(manifest.ManifestError):inf.build_report(campaign)


class Inference421Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(source_dir.cleanup)
        config, cluster, _ = inf.load_config(CONFIG_4_2_1)
        head = ready_probe(config, cluster, cluster['nodes'][0])
        packages = json.loads(head['container_probe']['stdout'])['packages']
        cls.canonical_inputs, cls.alternate_inputs = fixture_campaign_inputs(Path(source_dir.name), config, packages)
        cls.input_hashes = {name: manifest.sha256_file(path) for name, path in cls.canonical_inputs.items()}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.git = {'commit': 'a' * 40, 'dirty': False}
        captures = mock.patch.multiple(manifest, capture_git=mock.Mock(return_value=self.git),
                                       capture_nvidia=mock.Mock(return_value={}),
                                       capture_system=mock.Mock(return_value={}))
        captures.start()
        self.addCleanup(captures.stop)

    def test_plan_covers_three_ordered_blocks_without_execution(self):
        with (mock.patch.object(rt, 'execute', side_effect=AssertionError('process')),
              mock.patch.object(inf, 'write_new', side_effect=AssertionError('write'))):
            plan = inf.make_plan(CONFIG_4_2_1)

        self.assertEqual(plan['protocol']['protocol_id'], 'inference-4-2-1-v1')
        self.assertEqual([job['name'] for job in plan['jobs']], [
            f'b{block}-{model}-{group}'
            for block in (1, 2, 3) for model in MODEL_KEYS for group in ROSTERS
        ])
        self.assertEqual(plan['job_count'], 18)
        self.assertEqual(plan['cell_count'], 108)
        self.assertEqual(plan['measured_requests'], 3456)
        self.assertEqual(plan['warmup_requests'], 1728)
        for job in plan['jobs']:
            with self.subTest(job=job['name']):
                self.assertEqual(job['nodes'], ROSTERS[job['group']])
                server = job['server_command']
                self.assertEqual(int(server[server.index('--tensor-parallel-size') + 1]),
                                 len(job['nodes']))
                self.assertCountEqual([(c['input_tokens'], c['concurrency']) for c in job['cells']],
                                      [(length, load) for length in (512, 2048) for load in (1, 4, 8)])
                for cell in job['cells']:
                    for phase, count in (('measured', '32'), ('warmup', '16')):
                        command = cell['commands'][phase]
                        self.assertEqual(command[command.index('--num-prompts') + 1], count)

    def test_run_closes_complete_after_all_eighteen_jobs(self):
        launched = []
        workloads = self.root / 'workloads'
        workloads.mkdir()
        prepared = workloads / 'warmup.jsonl'
        prepared.write_text('{"prompt":"original"}\n', encoding='utf-8')
        prepared_hash = manifest.sha256_file(prepared)

        def complete_job(config, cluster, config_path, cluster_path, job, *args):
            if not launched:
                prepared.write_text('{"prompt":"changed during first job"}\n', encoding='utf-8')
            launched.append(job['name'])
            return {'job': job['name'], 'status': 'complete'}

        run_locally = partial(inf.run_campaign, results_root=self.root)
        write_new = inf.write_new

        def write_fixture(path, value):
            path = Path(path)
            if not path.is_relative_to(self.root):
                self.assertTrue(path.name.startswith('inference-preflight-'))
                path = self.root / path.name
            write_new(path, value)

        with (mock.patch.object(rt, 'preflight', return_value={'errors': [], 'nodes': []}),
              mock.patch.object(inf, 'prepare_workloads', return_value=workloads),
              mock.patch.object(inf, 'run_job', side_effect=complete_job),
              mock.patch.object(inf, 'write_new', side_effect=write_fixture),
              mock.patch.object(inf, 'run_campaign', wraps=run_locally) as run_campaign,
              mock.patch.object(inf.signal, 'signal'),
              redirect_stdout(io.StringIO())):
            self.assertEqual(inf.main(['run', '--execute', '--config', str(CONFIG_4_2_1)]), 0)

        directory, = (self.root / 'inference-campaign').iterdir()
        run_campaign.assert_called_once()
        plan = inf.read_json(directory / 'plan.json')
        campaign = inf.read_json(directory / 'campaign.json')
        marker = inf.read_json(directory / 'COMPLETE')
        self.assertEqual(launched, [job['name'] for job in plan['jobs']])
        self.assertEqual(len(launched), 18)
        self.assertEqual(campaign['status'], 'complete')
        self.assertEqual(campaign['pending_jobs'], [])
        self.assertEqual(set(campaign['input_hashes']), {*RUNNER_SOURCES, 'workload/warmup.jsonl'})
        self.assertEqual(campaign['input_hashes']['workload/warmup.jsonl'], prepared_hash)
        self.assertEqual(marker['campaign_sha256'], manifest.sha256_file(directory / 'campaign.json'))
        self.assertFalse((directory / 'PARTIAL').exists())

    def test_report_cannot_close_a_shortened_matrix_as_complete(self):
        protocol, _, _ = inf.load_config(CONFIG_4_2_1)
        protocol['measurement']['blocks'] = 0
        inf.write_new(self.root / 'campaign.json', {
            'campaign_id': 'empty-matrix', 'status': 'complete', 'protocol': protocol,
            'pending_jobs': [], 'runs': [], 'git': self.git,
        })
        inf.write_new(self.root / 'plan.json', {'protocol': protocol, 'jobs': []})
        inf.write_new(self.root / 'preflight.json', {})
        inf.write_new(self.root / 'COMPLETE', {
            name + '_sha256': manifest.sha256_file(self.root / (name + '.json'))
            for name in ('campaign', 'plan', 'preflight')
        })
        with self.assertRaisesRegex(inf.InferenceError, 'medición'):
            inf.build_report(self.root)

    def capture_head_launch(self, config, cluster, cluster_path, group):
        job = next(job for job in inf.jobs(config) if job['group'] == group)
        probes = {'nodes': [ready_probe(config, cluster, node) for node in cluster['nodes']]}
        # Capture the launch and stop before waiting for Ray or vLLM.
        denied = command_result(stderr='fixture stops after command construction', returncode=1)
        with (mock.patch.object(inf, 'stage'), mock.patch.object(inf, 'collect'),
              mock.patch.object(inf, 'cleanup_container', return_value={'status': 'confirmed_absent'}),
              mock.patch.object(rt, 'remote', return_value=denied) as remote,
              mock.patch.object(inf, 'wait_ray_head', side_effect=AssertionError('connection')) as wait_ray,
              redirect_stdout(io.StringIO())):
            inf.run_job(config, cluster, CONFIG_4_2_1, cluster_path, job, 'fixture',
                        probes, self.root / 'workloads', inf.time.monotonic() + 60, self.root)

        remote.assert_called_once()
        wait_ray.assert_not_called()
        node, argv = remote.call_args.args
        self.assertEqual(node['node_id'], 'pegasus')
        launch = argv[-1]
        before_server, server_line = launch.split('\nexec ', 1)
        barrier_line = before_server.split('\npython3 -c ', 1)[1]
        barrier = shlex.split(barrier_line.split(' > /work/logs/ray-membership.json', 1)[0])[0]
        server = shlex.split('exec ' + server_line)
        self.assertEqual(server[:3], ['exec', 'vllm', 'serve'])
        self.assertEqual(server[server.index('--tensor-parallel-size') + 1], str(len(ROSTERS[group])))
        return barrier

    def test_single_launch_barrier_accepts_only_pegasus_and_uses_tp_one(self):
        config, cluster, cluster_path = inf.load_config(CONFIG_4_2_1)
        pegasus = {'Alive': True, 'NodeID': 'a' * 56, 'Resources': {'GPU': 1},
                   'NodeManagerAddress': config['network']['node_ips']['pegasus']}
        octopus = dict(pegasus, NodeID='b' * 56, NodeManagerAddress=config['network']['node_ips']['octopus'])
        same_address = dict(pegasus, NodeID='c' * 56)
        same_id = dict(octopus, NodeID=pegasus['NodeID'])
        cases = {
            'single': [([pegasus], True), ([octopus], False), ([pegasus, octopus], False),
                       ([pegasus, same_address], False), ([pegasus, pegasus], False)],
            'pair': [([pegasus, octopus], True), ([pegasus, same_id], False),
                     ([pegasus, same_address, octopus], False)],
        }
        for group, membership_cases in cases.items():
            barrier = self.capture_head_launch(config, cluster, cluster_path, group)
            for members, accepted in membership_cases:
                ray = SimpleNamespace(init=mock.Mock(), nodes=mock.Mock(return_value=members),
                                      shutdown=mock.Mock())
                deadline = config['measurement']['startup_limit_seconds']
                clock = SimpleNamespace(monotonic=mock.Mock(side_effect=[0, 0, deadline + 1]),
                                        sleep=mock.Mock())
                with (self.subTest(group=group, members=members),
                      mock.patch.dict(sys.modules, {'ray': ray, 'time': clock}), redirect_stdout(io.StringIO())):
                    if accepted:
                        exec(barrier, {})
                        ray.shutdown.assert_called_once()
                    else:
                        with self.assertRaises(TimeoutError):
                            exec(barrier, {})

    def close_campaign_fixture(self, *, failed_job=None, wrong_roster=None, drifts=None):
        drifts = drifts or {}
        config, cluster, cluster_path = inf.load_config(CONFIG_4_2_1)
        preflight = {'status': 'ready_for_smoke', 'errors': [],
                     'nodes': [ready_probe(config, cluster, node) for node in cluster['nodes']]}
        campaign = self.root / 'campaign'
        campaign_id = 'fixture-4-2-1'
        plan = inf.make_plan(CONFIG_4_2_1)
        refs = []
        all_jobs = list(inf.jobs(config))
        for job in all_jobs:
            model = job['model']
            name, group = job['name'], job['group']
            failed = name == failed_job
            status = 'failed' if failed else 'complete'
            errors = ['fixture: server ran out of memory'] if failed else []
            request = {'job': job, 'protocol': config, 'campaign_id': campaign_id}
            drift = drifts.get(name)
            run_git = self.git
            if drift in ('commit', 'second_commit'):
                run_git = dict(self.git, commit=('b' if drift == 'commit' else 'd') * 40)
            tokenizer_revision = 'c' * 40 if drift == 'tokenizer' else model['tokenizer_revision']
            archived = {**self.canonical_inputs, 'cluster': cluster_path, 'protocol': CONFIG_4_2_1}
            if drift == 'wrong_profile':
                qwen = 'workload/qwen35-9b'
                archived[f'{qwen}/512/measured.jsonl'] = self.canonical_inputs[f'{qwen}/2048/measured.jsonl']
            if drift == 'workload':
                archived.update(self.alternate_inputs)
            inputs = archived.copy()
            if drift == 'missing_code_hash':
                del inputs['infraestructura/inference_worker.py']
            run_preflight = copy.deepcopy(preflight) if drift == 'image' else preflight
            if drift == 'image':
                pegasus = next(node for node in run_preflight['nodes'] if node['node_id'] == 'pegasus')
                pegasus['image'] = command_result(json.dumps('sha256:' + 'd' * 64) + ' "arm64"')
            with mock.patch.object(manifest, 'capture_git', return_value=run_git):
                path = manifest.start_run(
                    kind='inference-serving', model_id=model['id'], model_revision=model['revision'],
                    tokenizer_id=model['tokenizer_id'], tokenizer_revision=tokenizer_revision,
                    requested_config=request, requested_config_path=CONFIG_4_2_1,
                    inputs=inputs, results_root=self.root / 'runs', run_id=name,
                )
            for logical_name, source in archived.items():
                target = path / 'raw/inputs' / logical_name
                target.parent.mkdir(parents=True, exist_ok=True)
                if logical_name in self.canonical_inputs:
                    target.hardlink_to(source)  # All campaign inputs in the temporary fixture are immutable.
                else:
                    shutil.copyfile(source, target)
            inf.write_new(path / 'raw/preflight.json', run_preflight)
            cells = []
            if not failed:
                nodes = plan['jobs'][len(refs)]['nodes']
                if wrong_roster and name == wrong_roster[0]:
                    nodes = wrong_roster[1]
                write_fixture_roster(path, model, nodes, run_preflight)
                duration = {'four': 8, 'pair': 12 + 4 * job['block'], 'single': 48}[group]
                cells = [write_fixture_cell(path, model, cell, duration, drift) for cell in job['cells']]
            if drift != 'missing_assessment':
                inf.write_new(path / 'metrics/assessment.json', {'valid': not failed, 'cells': cells, 'errors': errors})
            manifest.finalize_run(path, status=status, errors=errors)
            refs.append({'job': name, 'group': group, 'block': job['block'], 'model_key': model['key'],
                         'run_dir': str(path), 'status': status, 'errors': errors,
                         'manifest_sha256': manifest.sha256_file(path / 'manifest.json')})
            if failed:
                break

        status = 'partial' if failed_job else 'complete'
        inf.write_new(campaign / 'campaign.json', {
            'campaign_id': campaign_id, 'status': status, 'protocol': config, 'git': self.git,
            'input_hashes': self.input_hashes,
            'runs': refs, 'pending_jobs': [job['name'] for job in all_jobs[len(refs):]],
        })
        inf.write_new(campaign / 'plan.json', plan)
        inf.write_new(campaign / 'preflight.json', preflight)
        inf.write_new(campaign / status.upper(), {
            stem + '_sha256': manifest.sha256_file(campaign / (stem + '.json'))
            for stem in ('campaign', 'plan', 'preflight')
        })
        return campaign

    def test_complete_report_has_one_row_per_model_block_and_load(self):
        campaign = self.close_campaign_fixture()
        output = self.root / 'report.md'
        self.assertEqual(inf.main(['report', str(campaign), '--output', str(output)]), 0)
        report = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
        self.assertEqual(len(report['cells']), 108)
        self.assertEqual(report['excluded_cells'], [])
        rows = report['paired_comparisons']
        self.assertCountEqual([
            (row['model'], row['block'], row['input_tokens'], row['concurrency']) for row in rows
        ], [
            (model, block, length, concurrency) for model in MODEL_KEYS for block in (1, 2, 3)
            for length in (512, 2048) for concurrency in (1, 4, 8)
        ])
        for row in rows:
            t2 = (16, 20, 24)[row['block'] - 1]
            self.assertEqual(row['s4_2'], t2 / 8)
            self.assertEqual(row['e4_2'], t2 / 16)
            self.assertEqual(row['s2_1'], 48 / t2)
            self.assertEqual(row['e2_1'], 24 / t2)
            self.assertEqual(row['s4_1'], 6)
            self.assertEqual(row['e4_1'], 1.5)
        markdown = output.read_text(encoding='utf-8')
        for label in ('S4/2', 'E4/2', 'S2/1', 'E2/1', 'S4/1', 'E4/1'):
            self.assertIn(label, markdown)

    def test_complete_report_checks_the_sealed_plan_against_the_protocol(self):
        campaign = self.close_campaign_fixture()
        plan_path = campaign / 'plan.json'
        closure_path = campaign / 'COMPLETE'
        plan = inf.read_json(plan_path)
        closure = inf.read_json(closure_path)
        for error in ('missing_cell', 'duplicate_coordinate', 'measured_total', 'warmup_command', 'override_prompts'):
            with self.subTest(error=error):
                changed = copy.deepcopy(plan)
                cells = changed['jobs'][-1]['cells']
                if error == 'missing_cell':
                    cells.pop()
                elif error == 'duplicate_coordinate':
                    cells[-1] = copy.deepcopy(cells[0])
                elif error == 'measured_total':
                    changed['measured_requests'] -= 32
                elif error == 'warmup_command':
                    argv = cells[-1]['commands']['warmup']
                    argv[argv.index('--num-prompts') + 1] = '32'
                else:
                    cells[-1]['commands']['measured'].append('--num-prompts=1')
                plan_path.write_text(json.dumps(changed))
                closure_path.write_text(json.dumps({**closure, 'plan_sha256': manifest.sha256_file(plan_path)}))
                with self.assertRaisesRegex(inf.InferenceError, 'plan'):
                    inf.build_report(campaign)

    def test_failed_single_keeps_4_2_without_borrowing_from_other_trials(self):
        campaign = self.close_campaign_fixture(failed_job='b2-qwen35-9b-single')
        report = inf.build_report(campaign)
        self.assertEqual(report['status'], 'partial')
        rows = report['paired_comparisons']
        self.assertEqual(len(rows), 18)
        previous = [row for row in rows if row['block'] == 1 and row['model'] == MODEL_KEYS[0]]
        self.assertEqual(len(previous), 6)
        self.assertTrue(all(row['s4_1'] == 6 and row['s2_1'] == 3 for row in previous))
        affected = [row for row in rows if row['block'] == 2]
        self.assertCountEqual([(row['model'], row['input_tokens'], row['concurrency']) for row in affected],
                              [(MODEL_KEYS[0], length, load) for length in (512, 2048) for load in (1, 4, 8)])
        for row in affected:
            self.assertEqual(row['s4_2'], 2.5)
            self.assertEqual(row['e4_2'], 1.25)
            for metric in ('s2_1', 'e2_1', 's4_1', 'e4_1'):
                self.assertIsNone(row.get(metric))

    def test_failed_single_without_assessment_preserves_verified_4_2(self):
        single = 'b1-qwen35-9b-single'
        campaign = self.close_campaign_fixture(failed_job=single, drifts={single: 'missing_assessment'})
        report = inf.build_report(campaign)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['runs'][-1]['status'], 'failed')
        self.assertFalse((Path(report['runs'][-1]['run_dir']) / 'metrics/assessment.json').exists())
        self.assertEqual(len(report['cells']), 12)
        self.assertCountEqual(report['paired_comparisons'], [
            {'model': MODEL_KEYS[0], 'block': 1, 'input_tokens': length, 'concurrency': load,
             's4_2': 2.0, 'e4_2': 1.0}
            for length in (512, 2048) for load in (1, 4, 8)
        ])

    def test_report_rejects_wrong_node_even_with_valid_roster_size_and_ranks(self):
        campaign = self.close_campaign_fixture(wrong_roster=('b1-qwen35-9b-single', ['octopus']),
                                               failed_job='b1-llama31-8b-four')
        with self.assertRaisesRegex(inf.InferenceError, 'roster|topolog'):
            inf.build_report(campaign)

    def assert_exclusions(self, report, expected):
        def coordinates(row):
            return (row['model_key'], row['block'], row['group'], row['input_tokens'], row['concurrency'])

        excluded = report['excluded_cells']
        self.assertCountEqual([coordinates(row) for row in excluded], expected)
        self.assertTrue(set(expected).isdisjoint(coordinates(row) for row in report['cells']))
        for row in excluded:
            reasons = row['reasons']
            self.assertIsInstance(reasons, list)
            self.assertTrue(reasons and all(isinstance(reason, str) and reason.strip() for reason in reasons))

    def test_report_excludes_wrong_client_concurrency_and_inconsistent_t32(self):
        campaign = self.close_campaign_fixture(failed_job='b1-llama31-8b-four',
                                               drifts={'b1-qwen35-9b-single': 'client'})
        report = inf.build_report(campaign)
        invalid = {(512, 1), (512, 4), (2048, 8)}
        self.assert_exclusions(report, [(MODEL_KEYS[0], 1, 'single', *cell) for cell in invalid])
        wrong_time = next(row for row in report['excluded_cells'] if row['input_tokens'] == 2048)
        self.assertEqual((wrong_time['t32_seconds'], wrong_time['duration_seconds']), (49, 48))
        self.assertEqual(len(report['cells']), 15)
        self.assertEqual(len(report['paired_comparisons']), 6)
        for row in report['paired_comparisons']:
            self.assertEqual(row['s4_2'], 2)
            if (row['input_tokens'], row['concurrency']) in invalid:
                self.assertIsNone(row.get('s2_1'))
                self.assertIsNone(row.get('s4_1'))
            else:
                self.assertEqual((row['s2_1'], row['s4_1']), (3, 6))

    def test_report_checks_current_profile_and_requires_recorded_source_hashes(self):
        campaign = self.close_campaign_fixture(failed_job='b2-qwen35-9b-four', drifts={
            'b1-qwen35-9b-single': 'wrong_profile',     # Its own 512-token JSONL changed.
            'b1-llama31-8b-pair': 'wrong_profile',     # Only the unused Qwen file changed.
            'b1-llama31-8b-single': 'missing_code_hash',
        })
        report = inf.build_report(campaign)
        self.assert_exclusions(report, [
            (model, 1, 'single', length, load) for model, length in (
                (MODEL_KEYS[0], 512), (MODEL_KEYS[1], 512), (MODEL_KEYS[1], 2048))
            for load in (1, 4, 8)
        ])
        self.assertEqual(len(report['cells']), 27)
        self.assertEqual(len(report['paired_comparisons']), 12)
        for row in report['paired_comparisons']:
            self.assertEqual(row['s4_2'], 2)
            expected = 6 if (row['model'], row['input_tokens']) == (MODEL_KEYS[0], 2048) else None
            self.assertEqual(row.get('s4_1'), expected)

    def test_report_keeps_only_compatible_pairs_when_trial_identity_changes(self):
        campaign = self.close_campaign_fixture(drifts={
            'b1-qwen35-9b-pair': 'commit',
            'b2-qwen35-9b-pair': 'image',
            'b3-qwen35-9b-pair': 'workload',
            'b1-llama31-8b-pair': 'commit',
            'b2-llama31-8b-pair': 'second_commit',
        })
        report = inf.build_report(campaign)
        self.assert_exclusions(report, [
            (model, block, 'pair', length, load) for model in MODEL_KEYS for block in (1, 2, 3)
            if (model, block) != (MODEL_KEYS[1], 3) for length in (512, 2048) for load in (1, 4, 8)
        ])
        rows = report['paired_comparisons']
        self.assertEqual(len(rows), 36)
        for row in rows:
            self.assertEqual(row['s4_1'], 6)
            self.assertEqual(row['e4_1'], 1.5)
            if (row['model'], row['block']) != (MODEL_KEYS[1], 3):
                for metric in ('s4_2', 'e4_2', 's2_1', 'e2_1'):
                    self.assertIsNone(row.get(metric), (row['block'], metric))
            else:
                self.assertIsNotNone(row.get('s4_2'))
                self.assertIsNotNone(row.get('s2_1'))
        # The archived Llama pair durations are 16, 20, 24 s. Only block 3 used the campaign commit.
        self.assertIn('| llama31-8b | 2 | 512 | 1 | 1 | 24.000', inf.markdown_report(report))
        for point in prepare_plot_data(report)['points']:
            if point['model_key'] == MODEL_KEYS[0]:
                self.assertNotIn('pair', point['groups'])
            else:
                throughput = point['groups']['pair']['output_tokens_per_second']
                self.assertEqual(throughput['blocks'], 1)
                self.assertAlmostEqual(throughput['median'], (32 * 128) / 24)

    def test_report_rejects_tokenizer_revision_outside_the_planned_trial(self):
        campaign = self.close_campaign_fixture(drifts={'b1-qwen35-9b-single': 'tokenizer'},
                                               failed_job='b1-llama31-8b-four')
        with self.assertRaisesRegex(inf.InferenceError, 'identidad|tokenizer'):
            inf.build_report(campaign)

    def test_two_smaller_runs_must_share_the_campaign_commit(self):
        campaign = self.close_campaign_fixture(failed_job='b2-qwen35-9b-four', drifts={
            'b1-qwen35-9b-four': 'commit',
            'b1-llama31-8b-pair': 'commit',
            'b1-llama31-8b-single': 'commit',
        })
        report = inf.build_report(campaign)
        rows = report['paired_comparisons']
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual((row['block'], row['model']), (1, 'qwen35-9b'))
            self.assertEqual(set(row) - {'block', 'model', 'input_tokens', 'concurrency'},
                             {'s2_1', 'e2_1'})
            self.assertEqual((row['s2_1'], row['e2_1']), (3.0, 1.5))


if __name__=='__main__':unittest.main()
