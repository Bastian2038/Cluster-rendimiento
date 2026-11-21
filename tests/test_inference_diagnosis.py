"""CPU-only integration fixtures for the isolated diagnosis campaign."""

import copy
import errno
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock

from infraestructura import inference as inf, inference_diag_protocol as diag, manifest
from infraestructura import inference_runtime as runtime
from infraestructura import inference_diag_phases as phases, inference_diag_traces as traces
from infraestructura import inference_diag_client as client
from infraestructura.inference_evidence import RUNNER_SOURCES, runner_sources
from infraestructura.inference_evidence import diagnostic_cell_issues
from infraestructura.inference_metrics import summarize_bench_result
from tests.test_inference import ready_probe, fixture_campaign_inputs, write_fixture_roster, command_result
from tests.test_inference_metrics import fixture

CONFIG = inf.REPO_ROOT / 'experimentos/configs/inferencia-diagnostico-2048-c8.json'


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config, self.cluster, self.cluster_path = inf.load_config(CONFIG)
        self.git = {'commit': 'a' * 40, 'dirty': False}
        patcher = mock.patch.multiple(manifest, capture_git=mock.Mock(return_value=self.git),
            capture_system=mock.Mock(return_value={}), capture_nvidia=mock.Mock(return_value={}))
        patcher.start(); self.addCleanup(patcher.stop)

    def test_plan_is_local_rotated_and_keeps_auxiliary_commands_separate(self):
        with (mock.patch.object(diag, 'validate_historical', side_effect=AssertionError('historical raw')),
              mock.patch.object(runtime, 'remote', side_effect=AssertionError('remote')),
              mock.patch.object(inf, 'write_new', side_effect=AssertionError('write'))):
            plan = inf.make_plan(CONFIG)
        self.assertEqual([plan[k] for k in ('job_count', 'cell_count', 'warmup_requests', 'measured_requests')],
                         [22, 78, 1248, 2496])
        self.assertEqual(plan['maximum_occupation_seconds'], 14400)
        self.assertEqual([j['group'] for j in plan['jobs'][:6]], ['four', 'pair', 'single'] * 2)
        for block, order in enumerate(('ABC', 'BCA', 'CAB')):
            for job in plan['jobs'][block*6:block*6+6]:
                self.assertEqual([(c['input_tokens'], c['concurrency']) for c in job['cells']],
                                 [diag.COORDINATES[k] for k in order])
                self.assertEqual(job['series'], 'primary')
                self.assertNotIn('--profiler-config', job['server_command'])
                self.assertTrue(all('mode' not in c for c in job['cells']))
        for job in plan['jobs'][18:]:
            self.assertEqual(job['block'], 4)
            self.assertIn('--profiler-config', job['server_command'])
            self.assertEqual([c['mode'] for c in job['cells']], ['off']*3 + ['on']*3)
            for cell in job['cells']:
                argv = cell['commands']['measured']
                self.assertEqual(argv[:3], ['python3', '-m', 'infraestructura.inference_diag_client'])
                self.assertEqual(argv[argv.index('--result-dir')+1], '/work/metrics/'+diag.cell_id(cell))
                self.assertEqual(cell['commands']['warmup'][:3], ['vllm', 'bench', 'serve'])
                ports = [arg for arg in argv if arg.startswith('--ipre-profile-port=')]
                self.assertEqual(ports, [f"--ipre-profile-port={self.config['runtime']['api_port']}"]
                                 if cell['mode'] == 'on' else [])
        for job in plan['jobs']:
            for cell in job['cells']:
                for command in cell['commands'].values():
                    self.assertEqual(command.count('--ready-check-timeout-sec'), 1)
                    self.assertEqual(command[command.index('--ready-check-timeout-sec')+1], '0')
        for protocol in ('inferencia-4-2.json', 'inferencia-4-2-1.json'):
            legacy = inf.make_plan(CONFIG.parent/protocol)
            for job in legacy['jobs']:
                for cell in job['cells']:
                    for command in cell['commands'].values():
                        self.assertNotIn('--ready-check-timeout-sec', command)
        inf._validate_planned_matrix(plan, self.config, list(inf.jobs(self.config)))
        tampered = copy.deepcopy(plan); tampered['jobs'][-1]['server_command'].pop()
        with self.assertRaises(inf.InferenceError):
            inf._validate_planned_matrix(tampered, self.config, list(inf.jobs(self.config)))
        self.assertEqual(runner_sources('inference-4-2-1-v1'), RUNNER_SOURCES)
        self.assertEqual(runner_sources(diag.PROTOCOL), RUNNER_SOURCES + diag.DIAGNOSTIC_SOURCES)

    def test_pinned_workload_rejects_both_warmup_and_measured_drift(self):
        hashes = self.config['diagnosis']['historical_jsonl_sha256']
        diag.verify_workloads(self.config, hashes)
        for phase in ('warmup', 'measured'):
            changed = hashes.copy()
            key = next(k for k in changed if k.endswith(phase+'.jsonl'))
            changed[key] = '0'*64
            with self.assertRaisesRegex(ValueError, 'historical'):
                diag.verify_workloads(self.config, changed)

    def closed_campaign(self, *, complete=False, contaminated=False):
        config = copy.deepcopy(self.config)
        preflight = {'errors': [], 'nodes': [ready_probe(config, self.cluster, n) for n in self.cluster['nodes']]}
        packages = json.loads(preflight['nodes'][0]['container_probe']['stdout'])['packages']
        inputs, _ = fixture_campaign_inputs(self.root/'inputs', config, packages)
        for name in diag.DIAGNOSTIC_SOURCES:
            path = self.root/'inputs/prepared'/name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# fixture\n')
            inputs[name] = path
        hashes = {k: manifest.sha256_file(v) for k, v in inputs.items()}
        config['diagnosis']['historical_campaign'] = '/absent/historical/campaign.json'
        config['diagnosis']['historical_jsonl_sha256'] = {k: v for k, v in hashes.items() if k.endswith('.jsonl')}
        history = self.root/'historical.json'
        inf.write_new(history, {'status': 'complete', 'errors': [], 'pending_jobs': [],
                               'protocol': config, 'input_hashes': hashes})
        config['diagnosis']['historical_campaign_sha256'] = manifest.sha256_file(history)
        protocol_path = self.root/'protocol.json'; inf.write_new(protocol_path, config)
        inputs.update({'protocol': protocol_path, 'cluster': self.cluster_path,
                       'diagnosis/historical-campaign.json': history})
        hashes = {k: manifest.sha256_file(v) for k, v in inputs.items()}
        campaign = self.root/'campaign'; campaign.mkdir()
        inf.write_new(campaign/'plan.json', inf.make_plan(protocol_path))
        inf.write_new(campaign/'preflight.json', preflight)
        references = []
        status = 'complete' if complete else 'failed'
        errors = [] if complete else ['scrape failed']
        for job in inf.jobs(config):
            model = job['model']
            path = manifest.start_run(kind='inference-serving', model_id=model['id'], model_revision=model['revision'],
                tokenizer_id=model['tokenizer_id'], tokenizer_revision=model['tokenizer_revision'],
                requested_config={'protocol': config, 'job': job, 'campaign_id': 'fixture'}, inputs=inputs,
                results_root=self.root/'runs')
            for name, source in inputs.items():
                target = path/'raw/inputs'/name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            inf.write_new(path/'raw/preflight.json', preflight)
            write_fixture_roster(path, model, inf.TOPOLOGIES[job['group']], preflight)
            start = datetime(2026, 11, 21, 20, tzinfo=timezone.utc)
            duration = {'four': 10, 'pair': 20, 'single': 40}[job['group']]
            end = start + timedelta(seconds=duration)
            logs = {}
            for rank, node in enumerate(inf.TOPOLOGIES[job['group']]):
                folder = path/'raw/nodes'/node/'logs'; folder.mkdir(parents=True, exist_ok=True)
                filenames = [f'worker-rank{rank:02d}.log'] + (['server.log'] if node == 'pegasus' else [])
                for filename in filenames: (folder/filename).write_text('')
                logs[node] = {filename: [0, 0] for filename in filenames}
                probe = next(p for p in preflight['nodes'] if p['node_id'] == node)
                samples = [dict(timestamp_utc=(start+timedelta(seconds=second)).isoformat(),
                    monotonic_ns=(second+1)*1_000_000_000, hostname=probe['hostname'],
                    gpus=[{'uuid': probe['gpu']['stdout'].split(',')[0]}], memory={'MemAvailable_kib': 9*1024**2})
                    for second in range(duration+1)]
                (folder/'telemetry.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
            cells = []
            for index, cell in enumerate(job['cells']):
                name = diag.cell_id(cell)
                cell_logs = copy.deepcopy(logs)
                if contaminated and job['name'] == 'b1-qwen35-9b-four' and index == 0:
                    text = b'compilation in measured window\n'
                    (path/'raw/nodes/pegasus/logs/server.log').write_bytes(text)
                    cell_logs['pegasus']['server.log'] = [0, len(text)]
                for phase, count in (('warmup', 16), ('measured', 32)):
                    raw = fixture(input_tokens=cell['input_tokens'], count=count,
                                  duration={'four': 10, 'pair': 20, 'single': 40}[job['group']])
                    raw.update(max_concurrency=cell['concurrency'], request_rate='inf', model_id=model['key'],
                               tokenizer_id=runtime.model_path(model), endpoint_type='openai')
                    inf.write_new(path/'raw/nodes/pegasus/metrics'/name/(phase+'.json'), raw)
                cells.append(dict(cell, cell=name, status='valid', client_returncode=0, client_timed_out=False,
                    measurement_started_at=start.isoformat(), measurement_ended_at=end.isoformat(),
                    **summarize_bench_result(raw, expected_prompt_tokens=cell['input_tokens'])))
                inf.write_new(path/'raw/diagnostic'/name/'execution-window.json', dict(cell=name,
                    measurement_started_at=start.isoformat(), measurement_ended_at=end.isoformat(), logs=cell_logs))
                ranks = [inf.read_json(path/p['evidence_path']) for p in manifest.load_run(path)['participants']]
                quality_issues = inf._cell_quality_issues(path, cells[-1], ranks)
                inf.write_new(path/'raw/diagnostic'/name/'quality.json', {'valid': not quality_issues, 'issues': quality_issues})
            inf.write_new(path/'metrics/assessment.json', {'valid': complete, 'cells': cells, 'errors': errors})
            manifest.finalize_run(path, status=status, errors=errors)
            references.append(dict(job=job['name'], group=job['group'], block=job['block'], model_key=model['key'],
                series=job['series'], run_dir=str(path), status=status, manifest_sha256=manifest.sha256_file(path/'manifest.json')))
        inf.write_new(campaign/'campaign.json', {'campaign_id': 'fixture', 'status': 'complete' if complete else 'partial', 'git': self.git,
            'protocol': config, 'input_hashes': hashes, 'runs': references, 'errors': errors, 'pending_jobs': []})
        self.seal(campaign, complete=complete)
        return campaign

    def seal(self, campaign, *, complete=False):
        inf.write_new(campaign/('COMPLETE' if complete else 'PARTIAL'), {key: manifest.sha256_file(campaign/name) for key, name in (
            ('campaign_sha256', 'campaign.json'), ('plan_sha256', 'plan.json'), ('preflight_sha256', 'preflight.json'))})

    def test_failed_diagnostics_preserve_all_primary_and_auxiliary_metrics_without_external_raw(self):
        campaign = self.closed_campaign()
        with mock.patch.object(diag, 'validate_historical', side_effect=AssertionError('external historical raw')):
            report = inf.build_report(campaign)
        self.assertEqual(len(report['cells']), 54)
        self.assertEqual(len(report['auxiliary_cells']), 24)
        self.assertEqual(len(report['paired_comparisons']), 18)
        self.assertTrue(all('mode' not in c and c['block'] <= 3 for c in report['cells']))
        self.assertTrue(all(c['mode'] in ('off', 'on') and c['block'] == 4 for c in report['auxiliary_cells']))
        self.assertTrue(all(p['block'] <= 3 and p['s4_2'] == 2 and p['s4_1'] == 4
                            for p in report['paired_comparisons']))
        self.assertEqual(report['excluded_cells'], [])
        self.assertFalse(any(c['diagnostic_valid'] for c in report['cells']))
        for cell in report['auxiliary_cells']:
            if cell['mode'] == 'on':
                self.assertFalse(cell['profile_control_valid'])
                self.assertFalse(cell['traces_valid'])
        with mock.patch.object(inf, 'diagnostic_cell_issues', side_effect=lambda p, m, c, r:
                               ['traces: missing rank'] if c.get('mode') == 'on' else []):
            partial = inf.build_report(campaign)
        on = [c for c in partial['auxiliary_cells'] if c['mode'] == 'on']
        self.assertEqual(len(on), 12)
        self.assertTrue(all(c['profile_control_valid'] and not c['traces_valid'] for c in on))

    def test_complete_rejects_missing_control_or_trace_evidence(self):
        campaign = self.closed_campaign(complete=True)
        for prefix in ('profile control:', 'profile coverage:', 'traces:'):
            with self.subTest(prefix=prefix), mock.patch.object(
                    inf, 'diagnostic_cell_issues', side_effect=lambda p, m, c, r:
                    [prefix+' incomplete'] if c.get('mode') == 'on' else []):
                with self.assertRaisesRegex(inf.InferenceError, 'evidencia diagnóstica incompleta'):
                    inf.build_report(campaign)
        original_read = inf.read_json
        def empty_assessment(path):
            return {'valid': True, 'cells': []} if Path(path).name == 'assessment.json' else original_read(path)
        with mock.patch.object(inf, 'read_json', side_effect=empty_assessment):
            with self.assertRaisesRegex(inf.InferenceError, 'exact nonempty cell matrix'):
                inf.build_report(campaign)
        with mock.patch.object(inf, 'diagnostic_cell_issues', return_value=[]):
            complete = inf.build_report(campaign)
        self.assertEqual(len(complete['cells']), 54)
        self.assertEqual(sum(c['mode'] == 'off' for c in complete['auxiliary_cells']), 12)
        self.assertEqual(sum(c['mode'] == 'on' for c in complete['auxiliary_cells']), 12)

    def test_partial_excludes_only_contaminated_cell_and_preserves_raw_t32(self):
        campaign = self.closed_campaign(contaminated=True)
        report = inf.build_report(campaign)
        self.assertEqual(len(report['raw_cells']), 78)
        self.assertEqual(len(report['cells']), 53)
        self.assertEqual(len(report['excluded_cells']), 1)
        rejected = report['excluded_cells'][0]
        self.assertEqual(rejected['cell'], '2048-c8')
        self.assertTrue(any('compilation' in reason for reason in rejected['reasons']))
        self.assertEqual(rejected['t32_seconds'], 10)
        paired = next(p for p in report['paired_comparisons'] if
                      (p['model'], p['block'], p['input_tokens'], p['concurrency']) == ('qwen35-9b', 1, 2048, 8))
        self.assertNotIn('s4_2', paired)
        self.assertNotIn('s4_1', paired)
        self.assertEqual(paired['s2_1'], 2)

    def test_client_profile_receipts_do_not_wrap_docker_and_preserve_native_before_failed_after(self):
        job = list(inf.jobs(self.config))[18]
        cell = dict(job['cells'][3]); directory = self.root/'run'; workspace = self.root/'workspace'
        nodes = [self.cluster['nodes'][0]]
        probes = {'pegasus': ready_probe(self.config, self.cluster, nodes[0])}
        name = diag.cell_id(cell)
        for phase, count in (('warmup', 16), ('measured', 32)):
            raw = fixture(input_tokens=2048, count=count)
            raw.update(max_concurrency=8, request_rate='inf', model_id=job['model']['key'],
                       tokenizer_id=runtime.model_path(job['model']), endpoint_type='openai')
            inf.write_new(workspace/'metrics'/name/(phase+'.json'), raw)
        inf.write_new(workspace/'metrics'/name/'client-events.json', {})
        control = {'valid': True, 'requires_server_shutdown': False}
        for action, sent, ack in (('start', 1, 2), ('stop', 3, 4)):
            control[action] = {'method': 'POST', 'path': f'/{action}_profile', 'http_status': 200, 'error': None,
                'sent': {'monotonic_ns': sent*10**9, 'utc': '2026-11-21T20:00:00+00:00'},
                'ack': {'monotonic_ns': ack*10**9, 'utc': '2026-11-21T20:00:01+00:00'}}
        inf.write_new(workspace/'metrics'/name/'profile-control.json', control)
        original_write = inf.write_new
        def receipt_failure(path, value):
            if Path(path).name == 'after.json':
                self.assertTrue((directory/'metrics'/f'{name}.json').is_file())
                self.assertTrue((directory/'raw/nodes/pegasus/metrics'/name/'measured.json').is_file())
                raise OSError('fixture receipt failure')
            original_write(path, value)
        before = {'pegasus': {'worker-rank00.log': 0, 'server.log': 0}}
        after = {'pegasus': {key: {'bounds': [0, 0], 'text': ''} for key in before['pegasus']}}
        with (mock.patch.object(phases, 'capture', return_value={'valid': False}) as capture,
              mock.patch.object(phases, 'phase_deltas', return_value={'valid': False}) as delta,
              mock.patch.object(traces, 'trace_window', side_effect=AssertionError('controller profiler')),
              mock.patch.object(traces, 'collect_traces', return_value={'valid': False}),
              mock.patch.object(inf, 'execution_logs', side_effect=[before, after]),
              mock.patch.object(inf, 'write_new', side_effect=receipt_failure),
              mock.patch.object(runtime, 'remote', return_value=command_result()) as remote):
            record, errors = inf._run_diagnostic_cell(self.config, job, cell, nodes, probes, str(workspace),
                'fixture', directory, time.monotonic()+300, [], [], [], {'pegasus': {}})
        self.assertEqual(remote.call_count, 2)
        self.assertEqual(record['status'], 'valid')
        self.assertNotIn('server_must_close', record)
        self.assertTrue(any('receipt failure' in e for e in errors))
        self.assertEqual(capture.call_args.args[-1], job['model']['key'])
        self.assertIn('measured', delta.call_args.kwargs)
        self.assertEqual(inf.read_json(directory/'raw/diagnostic'/name/'profile-control.json'), control)
        self.assertTrue((directory/'raw/diagnostic'/name/'profile-coverage.json').is_file())

    def test_uncertain_stop_explicit_stop_failure_and_cli_abort_close_server_without_next_warmup(self):
        for failure in ('unconfirmed', 'stop_log', 'cli_abort'):
            with self.subTest(failure=failure):
                config = copy.deepcopy(self.config)
                config['paths']['workspace'] = str(self.root/failure/'workspace')
                job = list(inf.jobs(config))[18]
                probes = {'nodes': [ready_probe(config, self.cluster, n) for n in self.cluster['nodes']]}
                ranks = []
                for rank, probe in enumerate(probes['nodes']):
                    packages = json.loads(probe['container_probe']['stdout'])['packages']
                    ranks.append(dict(node_id=probe['node_id'], rank=rank, hostname=probe['hostname'],
                        gpu_uuid=probe['gpu']['stdout'].split(',')[0], world_size=4, effective_tp=4,
                        effective_pp=1, dtype='torch.bfloat16', stage='load_model',
                        model_path=runtime.model_path(job['model']), tokenizer_path=runtime.model_path(job['model']),
                        versions={k: packages[k] for k in ('vllm', 'torch', 'ray')}))
                prepared = self.root/failure/'inputs'; prepared.mkdir(parents=True)
                inputs = {}
                for length in (512, 2048):
                    for filename, contents in (('metadata.json', '{"rows":[]}'), ('warmup.jsonl', ''), ('measured.jsonl', '')):
                        source = prepared/f'{length}-{filename}'; source.write_text(contents)
                        inputs[f"workload/{job['model']['key']}/{length}/{filename}"] = source
                workspace = Path(config['paths']['workspace'])/'fixture'/job['name']
                clients = []
                control = {'valid': True, 'requires_server_shutdown': False}
                for action, sent, ack in (('start', 1, 2), ('stop', 3, 4)):
                    control[action] = {'method': 'POST', 'path': f'/{action}_profile', 'http_status': 200,
                        'error': None, 'sent': {'monotonic_ns': sent*10**9, 'utc': '2026-11-21T20:00:00+00:00'},
                        'ack': {'monotonic_ns': ack*10**9, 'utc': '2026-11-21T20:00:01+00:00'}}
                if failure == 'unconfirmed':
                    control['stop']['ack'] = None; control['requires_server_shutdown'] = True
                def remote(node, argv, **kwargs):
                    if '--result-filename' not in argv: return command_result('container-id')
                    phase = argv[argv.index('--result-filename')+1].split('.')[0]
                    name = Path(argv[argv.index('--result-dir')+1]).name
                    cell = next(c for c in job['cells'] if diag.cell_id(c) == name)
                    clients.append((name, phase))
                    raw = fixture(input_tokens=cell['input_tokens'], count=16 if phase == 'warmup' else 32,
                                  model=job['model'], concurrency=cell['concurrency'])
                    inf.write_new(workspace/'metrics'/name/(phase+'.json'), raw)
                    if phase == 'measured':
                        inf.write_new(workspace/'metrics'/name/'client-events.json', {})
                        if cell['mode'] == 'on':
                            inf.write_new(workspace/'metrics'/name/'profile-control.json', control)
                            if failure == 'cli_abort': return command_result(returncode=1)
                    return command_result()
                def logs(nodes, ws, deadline, observed, offsets=None):
                    mapping = {r['node_id']: [f"worker-rank{r['rank']:02d}.log"] +
                               (['server.log'] if r['node_id'] == 'pegasus' else []) for r in observed}
                    if offsets is None: return {node: {name: 0 for name in names} for node, names in mapping.items()}
                    text = 'Failed to stop profiler\n' if failure == 'stop_log' and clients[-1][0].endswith('-on') else ''
                    return {node: {name: {'bounds': [0, len(text)], 'text': text} for name in names}
                            for node, names in mapping.items()}
                with (mock.patch.object(inf, '_experiment_inputs', return_value=inputs),
                      mock.patch.object(diag, 'verify_workloads'), mock.patch.object(inf, 'stage'),
                      mock.patch.object(inf, 'wait_ray_head'), mock.patch.object(inf, 'wait_ready'),
                      mock.patch.object(inf, '_rank_records', return_value=ranks),
                      mock.patch.object(inf, 'execution_logs', side_effect=logs),
                      mock.patch.object(phases, 'capture', return_value={'valid': False}),
                      mock.patch.object(phases, 'phase_deltas', return_value={'valid': False}),
                      mock.patch.object(traces, 'collect_traces', return_value={'valid': False}),
                      mock.patch.object(runtime, 'remote', side_effect=remote),
                      mock.patch.object(inf, 'cleanup_container', return_value={'status': 'confirmed_absent'}) as cleanup,
                      mock.patch.object(inf, 'collect'), redirect_stdout(io.StringIO())):
                    result = inf.run_job(config, self.cluster, CONFIG, self.cluster_path, job, 'fixture', probes,
                        prepared, time.monotonic()+300, self.root/failure/'runs')
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['failure_scope'], 'operational')
                self.assertEqual(len(clients), 8)
                self.assertEqual(clients[-1], ('2048-c8-on', 'measured'))
                self.assertGreaterEqual(cleanup.call_count, 4)
                run = Path(result['run_dir'])
                self.assertTrue((run/'FAILED').is_file())
                self.assertEqual(inf.read_json(run/'metrics/2048-c8-on.json')['t32_seconds'], 16)

    def test_campaign_continues_after_instrument_failures_and_excludes_cpu_preparation(self):
        workload = self.root/'workloads'; workload.mkdir()
        original_write = inf.write_new
        clock = {'seconds': 100.0}
        def history(*args):
            clock['seconds'] += 10
        def preflight(*args, **kwargs):
            clock['seconds'] += 20
            return {'nodes': [], 'errors': []}
        def prepare(*args):
            clock['seconds'] += 30
            return workload
        def write(path, value):
            path = Path(path)
            if not path.is_relative_to(self.root): path = self.root/path.name
            original_write(path, value)
        def failed_instrument(config, cluster, config_path, cluster_path, job, *args):
            # Validation/preflight/preparation consumed 60s before occupation.
            deadline = args[-2]
            self.assertEqual(deadline, 160 + 14100)
            clock['seconds'] += 10
            return dict(job=job['name'], status='failed', client_cells_complete=True, failure_scope='diagnostic')
        with (mock.patch.object(runtime, 'preflight', side_effect=preflight),
              mock.patch.object(inf, 'prepare_workloads', side_effect=prepare),
              mock.patch.object(inf, 'run_job', side_effect=failed_instrument) as run,
              mock.patch.object(inf, 'write_new', side_effect=write),
              mock.patch.object(diag, 'validate_historical', side_effect=history),
              mock.patch.object(diag, 'verify_workloads'),
              mock.patch.object(inf.time, 'monotonic', side_effect=lambda: clock['seconds']),
              redirect_stdout(io.StringIO())):
            closed = inf.run_campaign(CONFIG, results_root=self.root)
        campaign = inf.read_json(closed/'campaign.json')
        self.assertEqual(run.call_count, 22)
        self.assertEqual(len(campaign['runs']), 22)
        self.assertEqual(campaign['pending_jobs'], [])
        self.assertEqual(campaign['duration_seconds'], 220)
        self.assertEqual(campaign['effective_session_limit_seconds'], 14400)
        self.assertEqual(campaign['status'], 'partial')
        self.assertTrue((closed/'PARTIAL').is_file())

    def test_finalize_exception_after_failed_seal_keeps_real_attempt(self):
        config, inventory, cluster_path = inf.load_config()
        workload = self.root/'workload'; workload.mkdir()
        preflight = {'nodes': [ready_probe(config, inventory, n) for n in inventory['nodes']]}
        original_finalize = manifest.finalize_run
        def seal_then_raise(path, **kwargs):
            original_finalize(path, status='failed', errors=['fixture: changed inputs'])
            raise manifest.ManifestError('sealed FAILED')
        with (mock.patch.object(inf, 'stage', side_effect=RuntimeError('fixture stage failure')),
              mock.patch.object(inf, '_telemetry_valid', return_value=[]),
              mock.patch.object(manifest, 'finalize_run', side_effect=seal_then_raise),
              redirect_stdout(io.StringIO())):
            result = inf.run_job(config, inventory, inf.DEFAULT_CONFIG, cluster_path,
                next(inf.jobs(config)), 'fixture', preflight, workload, time.monotonic()+100, self.root)
        self.assertEqual(result['status'], 'failed')
        path = Path(result['run_dir'])
        self.assertEqual(manifest.load_run(path)['status'], 'failed')
        self.assertEqual(result['manifest_sha256'], manifest.sha256_file(path/'manifest.json'))

    def test_cleanup_log_enospc_still_attempts_stop_remove_and_verifies_exact_id(self):
        container_id = 'a'*64
        for confirmed in (True, False):
            with self.subTest(absence_confirmed=confirmed):
                replies = [command_result(container_id+' own'), command_result('logs'),
                           command_result(), command_result(),
                           command_result(stderr='No such object', returncode=1) if confirmed else command_result(container_id)]
                with (mock.patch.object(runtime, 'remote', side_effect=replies) as remote,
                      mock.patch.object(inf, 'write_new', side_effect=OSError(errno.ENOSPC, 'No space left on device')),
                      self.assertRaises(inf.CleanupError) as raised):
                    inf.cleanup_container(self.cluster['nodes'][0], 'own', self.root)
                calls = [call.args[1] for call in remote.call_args_list]
                self.assertIn(['docker', 'stop', '--time', '10', container_id], calls)
                self.assertIn(['docker', 'rm', container_id], calls)
                self.assertEqual(calls[-1], ['docker', 'inspect', '--type', 'container', container_id, '--format', '{{.Id}}'])
                receipt = raised.exception.receipt
                self.assertEqual(receipt['status'], 'failed')
                self.assertEqual(receipt['absence_confirmed'], confirmed)
                self.assertTrue(any('No space left' in issue for issue in receipt['errors']))
                self.assertTrue({'stop', 'remove', 'absence', 'write_logs'} <= receipt['timing_seconds'].keys())

    def test_launch_receipt_enospc_does_not_bypass_registered_server_cleanup(self):
        config, inventory, cluster_path = inf.load_config()
        workload = self.root/'workload'; workload.mkdir()
        preflight = {'nodes': [ready_probe(config, inventory, n) for n in inventory['nodes']]}
        original_write = inf.write_new; container_id = 'b'*64
        commands = []
        def write(path, data):
            if Path(path).name == 'launch.json':
                raise OSError(errno.ENOSPC, 'No space left on device')
            original_write(path, data)
        def remote(node, argv, **kwargs):
            commands.append(argv)
            if argv[:2] == ['docker', 'run']:
                raise RuntimeError('fixture: stop after registering own launch')
            if argv[:2] == ['docker', 'inspect']:
                if argv[4] == container_id: return command_result(stderr='No such object', returncode=1)
                return command_result(container_id+' '+argv[4])
            return command_result()
        with (mock.patch.object(inf, 'stage'), mock.patch.object(runtime, 'remote', side_effect=remote),
              mock.patch.object(inf, 'write_new', side_effect=write), mock.patch.object(inf, 'collect'),
              mock.patch.object(inf, '_telemetry_valid', return_value=[]), redirect_stdout(io.StringIO())):
            result = inf.run_job(config, inventory, inf.DEFAULT_CONFIG, cluster_path,
                next(inf.jobs(config)), 'fixture-enospc', preflight, workload, time.monotonic()+100, self.root)
        self.assertIn(['docker', 'stop', '--time', '10', container_id], commands)
        self.assertIn(['docker', 'rm', container_id], commands)
        self.assertIn(['docker', 'inspect', '--type', 'container', container_id, '--format', '{{.Id}}'], commands)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('launch receipt' in issue and 'No space left' in issue for issue in result['errors']))
        run = Path(result['run_dir'])
        self.assertTrue((run/'FAILED').exists())
        cleanup = inf.read_json(run/'raw/cleanup.json')
        self.assertTrue(cleanup[0]['absence_confirmed'])

    def test_generic_collector_never_visits_private_traces(self):
        self.assertNotIn("('evidence','metrics','logs','traces')", inf.COLLECT)
        self.assertIn("('evidence','metrics','logs')", inf.COLLECT)

    def test_trace_control_uses_sent_ack_window_and_allows_partial_batch_coverage(self):
        cell = {'input_tokens': 2048, 'concurrency': 8, 'mode': 'on'}
        dest = self.root/'raw/diagnostic'/diag.cell_id(cell)
        model = self.config['models'][0]
        def mark(ns):
            return {'monotonic_ns': ns, 'utc': '2026-11-21T20:00:00+00:00'}
        control = {'valid': True, 'opened': mark(500_000_000), 'closed': mark(90_000_000_000)}
        for action, sent, ack in (('start', 1_000_000_000, 2_000_000_000),
                                   ('stop', 60_000_000_000, 61_000_000_000)):
            control[action] = {'method': 'POST', 'path': f'/{action}_profile',
                               'sent': mark(sent), 'ack': mark(ack), 'http_status': 200, 'error': None}
        for name, value in (('before.json', {}), ('after.json', {}), ('phases.json', {'valid': True}),
                            ('client-validation.json', {'valid': True}), ('traces.json', {'valid': True}),
                            ('profile-coverage.json', {'valid': True, 'count': 1}),
                            ('profile-control.json', control)):
            inf.write_new(dest/name, value)
        raw_dir = self.root/'raw/nodes/pegasus/metrics'/diag.cell_id(cell)
        inf.write_new(raw_dir/'client-events.json', {}); inf.write_new(raw_dir/'measured.json', {})
        from infraestructura import inference_diag_client as client
        with (mock.patch.object(phases, 'phase_deltas', return_value={'valid': True}),
              mock.patch.object(client, 'verify_events', return_value={'valid': True}),
              mock.patch.object(client, 'profile_coverage', return_value={'valid': True, 'count': 1}),
              mock.patch.object(traces, 'validate_traces', return_value={'valid': True}) as verify_traces):
            self.assertEqual(diagnostic_cell_issues(self.root, model, cell, []), [])
            for action, field, ns in (('stop', 'ack', 61_000_000_001),
                                       ('stop', 'sent', 1_000_000_000),
                                       ('start', 'sent', True)):
                tampered = copy.deepcopy(control); tampered[action][field]['monotonic_ns'] = ns
                (dest/'profile-control.json').write_text(json.dumps(tampered))
                issues = diagnostic_cell_issues(self.root, model, cell, [])
                self.assertEqual(len(issues), 1)
                self.assertTrue(issues[0].startswith('profile control:'))
            self.assertEqual(verify_traces.call_count, 4)
            # Valid acknowledgements survive missing/invalid Kineto evidence.
            (dest/'profile-control.json').write_text(json.dumps(control))
            verify_traces.return_value = {'valid': False}
            issues = diagnostic_cell_issues(self.root, model, cell, [])
            self.assertEqual(len(issues), 1)
            self.assertTrue(issues[0].startswith('traces:'))
            # Both failures are reported independently, without short circuit.
            (dest/'profile-control.json').write_text(json.dumps({'valid': False}))
            issues = diagnostic_cell_issues(self.root, model, cell, [])
            self.assertTrue(any(i.startswith('profile control:') for i in issues))
            self.assertTrue(any(i.startswith('traces:') for i in issues))

    def test_real_profile_coverage_requires_measured_overlap_and_same_client_clock(self):
        from tests.test_inference_diag_phases import detailed_result, observed_events, profile_receipt
        raw = detailed_result(); events = observed_events(raw)
        cell = {'input_tokens': 2048, 'concurrency': 8, 'mode': 'on'}
        dest = self.root/'raw/diagnostic'/diag.cell_id(cell)
        raw_dir = self.root/'raw/nodes/pegasus/metrics'/diag.cell_id(cell)
        inf.write_new(raw_dir/'measured.json', raw); inf.write_new(raw_dir/'client-events.json', events)
        for name, value in (('before.json', {}), ('after.json', {}), ('phases.json', {'valid': True}),
                            ('client-validation.json', client.verify_events(events, raw)), ('traces.json', {'valid': True})):
            inf.write_new(dest/name, value)
        with (mock.patch.object(phases, 'phase_deltas', return_value={'valid': True}),
              mock.patch.object(traces, 'validate_traces', return_value={'valid': True})):
            for case, stop in (('overlap', 101.98), ('before_requests', 100.95), ('different_clock', 101.98)):
                control = profile_receipt(events, raw, 100.9, stop)
                for action in ('start', 'stop'):
                    control[action].update(method='POST', path=f'/{action}_profile')
                    for mark in ('sent', 'ack'): control[action][mark]['utc'] = '2026-11-21T20:00:00+00:00'
                if case == 'different_clock':
                    control['client_clock'] = dict(control['client_clock'], pid=control['client_clock']['pid']+1)
                coverage = client.profile_coverage(control, events, raw)
                (dest/'profile-control.json').write_text(json.dumps(control))
                (dest/'profile-coverage.json').write_text(json.dumps(coverage))
                issues = diagnostic_cell_issues(self.root, self.config['models'][0], cell, [])
                self.assertFalse(any(i.startswith('profile control:') for i in issues))
                self.assertEqual(any(i.startswith('profile coverage:') for i in issues), case != 'overlap')
                self.assertFalse(any(i.startswith('traces:') for i in issues))


if __name__ == '__main__':
    unittest.main()
