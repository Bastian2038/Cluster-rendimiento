"""CPU fixtures for orchestration, fail-closed readiness and paired reports."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from infraestructura import inference as inf
from infraestructura import inference_runtime as rt
from infraestructura import manifest
from infraestructura.inference_metrics import summarize_bench_result
from tests.test_inference_metrics import fixture as metrics_fixture


def command_result(stdout='', stderr='', returncode=0):
    return {'stdout':stdout,'stderr':stderr,'returncode':returncode,'timed_out':False}


def ready_probe(config, cluster_config, node):
    index=cluster_config['groups']['four'].index(node['node_id'])
    packages={'vllm':'0.24.0+nv','ray':'2.48.0','torch':'2.13.0','transformers':'5.6.1',
              'tokenizers':'0.22.2','huggingface-hub':'1.24.0','apache-tvm-ffi':'0.1.9'}
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
                     mock.patch.object(manifest,'capture_nvidia',return_value={}):
                    directory=manifest.start_run(kind='inference-serving',model_id=model['id'],
                        model_revision=model['revision'],tokenizer_id=model['id'],tokenizer_revision=model['revision'],
                        requested_config={'job':job,'protocol':self.config,'campaign_id':'fixture'},
                        inputs={'workload/fixture':workload},results_root=root/'raw')
                people=[]
                for rank,node_id in enumerate(nodes):
                    person={'node_id':node_id,'hostname':f'host-{rank}','gpu_uuid':f'GPU-{rank}','rank':rank}
                    evidence=f'raw/rank{rank}.json';inf.write_new(directory/evidence,person)
                    people.append(dict(person,evidence_path=evidence))
                manifest.register_participants(directory,people)
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
            self.assertEqual(len(report['paired_comparisons']),1)
            self.assertEqual(report['paired_comparisons'][0]['s4_2'],.5)
            self.assertEqual(report['paired_comparisons'][0]['e4_2'],.25)
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


if __name__=='__main__':unittest.main()
