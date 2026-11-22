"""CPU-only fixtures for reservation, immediate archives and campaign sealing."""

import copy
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from infraestructura import inference as inf, inference_diag_phases as phases, manifest
from infraestructura import serving_capacity_client as client, serving_capacity_protocol as protocol
from infraestructura import serving_capacity_runner as runner
from infraestructura.inference_evidence import RUNNER_SOURCES, runner_sources


def dump(path, value):
    inf.write_new(path, value)


def workload(root, model):
    for length in (512, 2048):
        profile = root / model['key'] / str(length)
        rows = []; hashes = {}
        for phase, count in (('warmup', 16), ('measured', 32)):
            prompts = []
            for index in range(count):
                prompt = f'{length}-{phase}-{index}'
                digest = hashlib.sha256(prompt.encode()).hexdigest()
                prompts.append({'prompt': prompt, 'output_tokens': 128})
                rows.append({'phase': phase, 'prompt_sha256': digest, 'prompt_tokens': length,
                             'output_tokens': 128, 'token_ids': [length]})
            profile.mkdir(parents=True, exist_ok=True)
            path = profile / (phase + '.jsonl')
            path.write_text(''.join(json.dumps(row) + '\n' for row in prompts))
            hashes[phase] = manifest.sha256_file(path)
        dump(profile / 'metadata.json', {'schema_version': 1, 'input_tokens': length, 'output_tokens': 128,
            'model_id': model['id'], 'model_revision': model['revision'],
            'tokenizer_id': model['tokenizer_id'], 'tokenizer_revision': model['tokenizer_revision'],
            'rows': rows, 'files_sha256': hashes})
        dump(profile / 'reader-validation.json', {'fixture': True})


class AtomicWriteTests(unittest.TestCase):
    def test_dump_interruption_never_publishes_partial_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); path = directory / 'receipt.json'
            def interrupted(data, stream, **kwargs):
                stream.write('{"partial":')
                stream.flush()
                self.assertFalse(path.exists())
                raise KeyboardInterrupt('during dump')
            with mock.patch.object(inf.json, 'dump', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    inf.write_new(path, {'complete': True})
            self.assertEqual(list(directory.iterdir()), [])

    def test_existing_destination_survives_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); path = directory / 'receipt.json'
            inf.write_new(path, {'original': True})
            original = path.read_bytes()
            with mock.patch.object(inf.json, 'dump', side_effect=AssertionError('serialized existing destination')):
                with self.assertRaises(FileExistsError):
                    inf.write_new(path, object())
            with self.assertRaises(FileExistsError):
                inf.write_new(path, {'replacement': True})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(directory.iterdir()), [path])

    def test_temporary_name_allows_maximum_destination_basename(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / ('x' * os.pathconf(directory, 'PC_NAME_MAX'))
            inf.write_new(path, {'complete': True})
            self.assertEqual(inf.read_json(path), {'complete': True})

    def test_concurrent_destination_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); path = directory / 'receipt.json'
            real_link = inf.os.link
            def raced(source, target):
                Path(target).write_bytes(b'concurrent writer\n')
                return real_link(source, target)
            with mock.patch.object(inf.os, 'link', side_effect=raced):
                with self.assertRaises(FileExistsError):
                    inf.write_new(path, {'replacement': True})
            self.assertEqual(path.read_bytes(), b'concurrent writer\n')
            self.assertEqual(list(directory.iterdir()), [path])

    def test_broken_symlink_is_existing_before_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'receipt.json'
            path.symlink_to('missing-target')
            with mock.patch.object(inf.json, 'dump', side_effect=AssertionError('serialization')):
                with self.assertRaises(FileExistsError):
                    inf.write_new(path, object())
            self.assertTrue(path.is_symlink())

    def test_historical_bytes_and_normal_umask_permissions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'receipt.json'
            value = {'z': 'á', 'a': [1, None, False]}
            previous = os.umask(0o027)
            try:
                inf.write_new(path, value)
            finally:
                os.umask(previous)
            expected = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
            self.assertEqual(path.read_bytes(), expected.encode('utf-8'))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)


class PointFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.cluster, self.cluster_path = protocol.load_config()
        self.model = self.config['models'][0]
        self.job = copy.deepcopy(next(protocol.jobs(self.config)))
        self.job['group'] = 'single'
        self.job['cells'] = [{'rate_rps': .25}, {'rate_rps': .5}]
        self.directory = self.root / 'run'
        self.workspace = self.root / 'workspace'
        self.workloads = self.directory / 'raw/inputs/workload'
        workload(self.workloads, self.model)
        self.clock = 100.0
        self.origin = datetime(2026, 11, 22, tzinfo=timezone.utc)
        self.node = inf.cluster.selected_nodes(self.cluster, 'single')[0]
        self.ranks = [{'node_id': 'pegasus', 'rank': 0, 'world_size': 1,
                       'hostname': self.node['expected_hostname'], 'gpu_uuid': 'GPU-fixture'}]
        self.active = []
        self.health_calls = []
        self.commands = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(manifest, 'capture_nvidia', side_effect=AssertionError('real GPU capture')))
        self.stack.enter_context(mock.patch.object(runner.time, 'monotonic', side_effect=lambda: self.clock))
        self.stack.enter_context(mock.patch.object(manifest, 'utc_now', side_effect=self.utc))
        self.stack.enter_context(mock.patch.object(runner, '_health', side_effect=self.health))
        self.stack.enter_context(mock.patch.object(inf, 'execution_logs', side_effect=self.logs))
        self.stack.enter_context(mock.patch.object(inf, 'container_command', side_effect=lambda *a, **k: ['docker', 'run', *a[5]]))
        self.stack.enter_context(mock.patch.object(runner.runtime, 'remote', side_effect=self.remote))

    def utc(self):
        return (self.origin + timedelta(seconds=self.clock)).isoformat()

    def health(self, config, model, deadline):
        self.clock += .01
        now = self.clock
        snapshot = {'valid': True, 'drained': True, 'http_status': 200,
            'health_http_status': 200,
            'model_key': model['key'], 'port': config['runtime']['api_port'],
            'models': {'data': [{'id': model['key']}]},
            'body': '\n'.join([
                '# TYPE vllm:num_requests_running gauge',
                '# TYPE vllm:num_requests_waiting gauge',
                f'vllm:num_requests_running{{model_name="{model["key"]}",engine="0"}} 0',
                f'vllm:num_requests_waiting{{model_name="{model["key"]}",engine="0"}} 0',
            ]),
            'started_monotonic_s': now, 'finished_monotonic_s': now,
            'process_identity': {'source': 'listener-procfs', 'boot_id': 'fixture-boot',
                'pid': 1, 'start_ticks': 2, 'socket_inode': 3, 'port': config['runtime']['api_port'],
                'checked_before_monotonic_s': now, 'checked_after_monotonic_s': now}}
        self.health_calls.append(snapshot)
        return snapshot

    def logs(self, nodes, workspace, deadline, ranks, offsets=None):
        files = ('worker-rank00.log', 'server.log')
        return {'pegasus': {name: 0 if offsets is None else {'bounds': [0, 0], 'text': ''} for name in files}}

    def remote(self, node, command, **kwargs):
        self.assertGreater(kwargs['timeout'], 0)
        self.commands.append(command)
        phase = command[command.index('--phase') + 1]
        rate = float(command[command.index('--rate') + 1])
        prompts = client.expected_prompts(self.workloads / self.model['key'], phase,
                                         self.config['measurement'][phase + '_requests'])
        rows = []
        for index, prompt in enumerate(prompts):
            scheduled = index / rate
            rejected = phase == 'measured' and index == 2
            rows.append({'index': index, 'input_tokens': prompt['input_tokens'],
                'prompt_sha256': prompt['prompt_sha256'], 'scheduled_s': scheduled,
                'sent_s': scheduled, 'first_token_s': None if rejected else scheduled + 1,
                'end_s': scheduled + 2, 'status': 'rejected' if rejected else 'ok',
                'output_tokens': 0 if rejected else 128})
            if not rejected:
                token_ids = list(range(128))
                rows[-1].update(last_token_s=scheduled + 1.9, stream_done_s=scheduled + 1.95,
                    response_end_s=scheduled + 2, http_status=200, finish_reason='length',
                    server_input_tokens=prompt['input_tokens'], output_tokens=len(token_ids))
        path = self.workspace / 'metrics' / protocol.point_name(rate) / (phase + '.json')
        dump(path, {'schema_version': 1, 'model': self.model['key'], 'phase': phase,
                    'rate_rps': rate, 'rows': rows})
        self.clock += len(rows) / rate + 4
        return {'returncode': 0, 'timed_out': False, 'stdout': '', 'stderr': ''}

    def point(self, coordinate=None):
        return runner.run_point(self.config, self.job, coordinate or self.job['cells'][0],
            [self.node], {'pegasus': {}}, str(self.workspace), 'fixture-campaign', self.directory,
            100000, self.active, self.ranks)

    def telemetry(self):
        directory = self.directory / 'raw/nodes/pegasus/logs'
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('worker-rank00.log', 'server.log'):
            (directory / name).write_text('')
        samples = [{'timestamp_utc': (self.origin + timedelta(seconds=t)).isoformat(),
                    'monotonic_ns': t * 10**9, 'hostname': self.node['expected_hostname'],
                    'gpus': [{'uuid': 'GPU-fixture'}], 'memory': {'MemAvailable_kib': 20 * 1024**2}}
                   for t in range(100, int(self.clock) + 2)]
        (directory / 'telemetry.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in samples))

    def test_archives_both_phases_before_next_point_and_keeps_errors(self):
        cell = self.point()
        self.assertFalse(self.active)
        self.assertEqual(cell['offered'], 128)
        self.assertEqual(cell['status_counts']['rejected'], 1)
        self.assertEqual(cell['compliant'], 127)
        native = self.directory / 'raw/nodes/pegasus/metrics/rate-0p25'
        self.assertTrue(all((native / (phase + '.json')).is_file() for phase in ('warmup', 'measured')))
        self.assertEqual(len(self.health_calls), 4)
        self.telemetry()
        cells, errors = runner.assess_job(self.directory, self.config, self.job, self.ranks)
        self.assertFalse(errors)
        self.assertEqual([c['status'] for c in cells], ['valid'])
        # A later failed point cannot discard the complete first point.
        with mock.patch.object(runner.runtime, 'remote', side_effect=OSError('transport')):
            with self.assertRaises(OSError):
                self.point(self.job['cells'][1])
        self.assertEqual(inf.read_json(self.directory / 'metrics/rate-0p25.json'), cell)
        self.assertTrue((native / 'measured.json').is_file())

    def test_changed_native_raw_is_detected_during_collection(self):
        self.point()
        source = self.workspace / 'metrics/rate-0p25/measured.json'
        source.write_text('{}')
        with self.assertRaisesRegex(inf.InferenceError, 'cambió'):
            inf._archive_client_file(self.directory, self.workspace, 'rate-0p25', 'measured.json')

    def test_undrained_warmup_stops_before_measurement(self):
        original = self.health
        calls = []
        def bad(*args):
            result = original(*args)
            calls.append(result)
            if len(calls) == 2:
                result['drained'] = False
            return result
        with mock.patch.object(runner, '_health', side_effect=bad):
            with self.assertRaisesRegex(inf.InferenceError, 'drenado'):
                self.point()
        self.assertEqual(len(self.commands), 1)
        self.assertTrue((self.directory / 'raw/nodes/pegasus/metrics/rate-0p25/warmup.json').is_file())

    def test_unhealthy_engine_blocks_client_even_with_zero_gauges(self):
        snapshot = self.health(self.config, self.model, 100000)
        snapshot['health_http_status'] = 503
        with mock.patch.object(runner, '_health', return_value=snapshot):
            with self.assertRaisesRegex(inf.InferenceError, '/health'):
                self.point()
        self.assertFalse(self.commands)

    def test_write_failure_leaves_client_registered_for_lifecycle_cleanup(self):
        original = inf.write_new
        def fail_window(path, value):
            if str(path).endswith('warmup-window.json'):
                raise OSError('disk full')
            original(path, value)
        with mock.patch.object(inf, 'write_new', side_effect=fail_window):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.point()
        self.assertEqual(len(self.active), 1)
        self.assertTrue((self.workspace / 'metrics/rate-0p25/warmup.json').is_file())

    def test_missing_raw_preserves_window_returncode_and_stderr(self):
        with mock.patch.object(runner.runtime, 'remote', return_value={
                'returncode': 7, 'timed_out': False, 'stdout': '', 'stderr': 'client failed'}):
            with self.assertRaises(FileNotFoundError):
                self.point()
        window = inf.read_json(self.directory / 'raw/serving-capacity/rate-0p25/warmup-window.json')
        result = inf.read_json(self.directory / 'logs/rate-0p25-warmup.json')
        self.assertEqual(window['client_returncode'], 7)
        self.assertEqual(result['stderr'], 'client failed')

    def test_client_oserror_is_not_reported_as_timeout(self):
        self.client_cut(OSError('transport failed'), None)

    def test_client_interrupt_is_not_reported_as_timeout(self):
        self.client_cut(KeyboardInterrupt('client interrupted'), None)

    def test_client_timeout_error_is_reported_as_timeout(self):
        self.client_cut(TimeoutError('client deadline'), True)

    def test_client_subprocess_timeout_is_reported_as_timeout(self):
        self.client_cut(subprocess.TimeoutExpired(['fixture'], 1), True)

    def client_cut(self, error, expected_timeout):
        with mock.patch.object(runner.runtime, 'remote', side_effect=error):
            with self.assertRaises(type(error)):
                self.point()
        window = inf.read_json(self.directory / 'raw/serving-capacity/rate-0p25/warmup-window.json')
        result = inf.read_json(self.directory / 'logs/rate-0p25-warmup.json')
        self.assertIs(window['client_timed_out'], expected_timeout)
        self.assertIs(result['timed_out'], expected_timeout)
        self.assertEqual(window['client_error_type'], type(error).__name__)
        self.assertEqual(window['client_error_reason'], str(error))
        self.assertEqual(result['error_type'], type(error).__name__)
        self.assertEqual(result['error_reason'], str(error))
        self.assertEqual(len(self.active), 1)

    def test_occupation_stops_before_collection_and_analysis(self):
        record, _, _, _ = self.lifecycle(collection_delay=100)
        occupation = runner.require_occupation(self.directory)
        self.assertTrue(occupation['removal_confirmed'])
        self.assertLess(occupation['ended_monotonic_s'], self.clock)
        self.assertEqual(record['occupation'], occupation)
        runner.require_tokenization(self.directory, self.config)
        runner.require_point_quality(self.directory, 'rate-0p25')

    def test_late_retirement_forces_failed_subrun(self):
        record, _, _, _ = self.lifecycle(expired=True)
        self.assertEqual(record['status'], 'failed')
        self.assertTrue(record['occupation']['limit_exceeded'])

    def test_unconfirmed_retirement_keeps_occupation_open(self):
        record, _, _, _ = self.lifecycle(unconfirmed=True)
        self.assertEqual(record['status'], 'failed')
        self.assertFalse(record['occupation']['removal_confirmed'])
        self.assertIsNone(record['occupation']['ended_at'])
        self.assertIsNone(record['occupation']['ended_monotonic_s'])

    def test_bad_telemetry_invalidates_saved_point(self):
        self.point()
        cells, errors = runner.assess_job(self.directory, self.config, self.job, self.ranks)
        self.assertTrue(errors)
        self.assertEqual(cells[0]['status'], 'invalid')

    def test_lifecycle_reuses_ranks_tokenizer_collection_and_cleanup(self):
        result, cleanup, final, tokenize = self.lifecycle()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(cleanup.call_count, 1)
        self.assertEqual(final.call_args.kwargs['status'], 'complete')
        self.assertEqual(tokenize.call_count, 96)
        assessment = inf.read_json(self.directory / 'metrics/assessment.json')
        self.assertTrue(assessment['valid'])
        self.assertTrue((self.directory / 'raw/participants/rank-0.json').is_file())
        self.assertTrue((self.directory / 'raw/inputs/reference/campaign.json').is_file())

    def test_lifecycle_cleans_client_and_server_after_receipt_write_failure(self):
        result, cleanup, final, _ = self.lifecycle(fail_write=True)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(cleanup.call_count, 2)
        self.assertEqual(final.call_args.kwargs['status'], 'failed')
        self.assertTrue((self.directory / 'raw/nodes/pegasus/metrics/rate-0p25/warmup.json').is_file())

    def test_verified_seal_rechecks_native_points_and_rejects_tampering(self):
        record, _, _, _ = self.lifecycle()
        run = manifest.load_run(self.directory)
        hashes = {item['name']: item['sha256'] for item in run['inputs']}
        with mock.patch.object(inf, '_observed_topology', return_value=({}, {}, True)):
            verified = runner._verified_record(record, self.job, self.config, 'fixture-campaign', hashes)
        self.assertEqual(verified['valid_cells'], ['rate-0p25'])
        source = self.directory / 'raw/nodes/pegasus/metrics/rate-0p25/measured.json'
        source.write_text('{}')
        with self.assertRaises(manifest.ManifestError):
            runner._verified_record(record, self.job, self.config, 'fixture-campaign', hashes)

    def test_failed_job_preserves_a_quality_verified_complete_point(self):
        record, _, _, _ = self.lifecycle(fail_after_point=True)
        self.assertEqual(record['status'], 'failed')
        run = manifest.load_run(self.directory)
        hashes = {item['name']: item['sha256'] for item in run['inputs']}
        with mock.patch.object(inf, '_observed_topology', return_value=({}, {}, True)):
            verified = runner._verified_record(record, self.job, self.config, 'fixture-campaign', hashes)
        self.assertEqual(verified['valid_cells'], ['rate-0p25'])
        self.assertTrue((self.directory / 'FAILED').is_file())
        self.assertEqual(len(inf.read_json(self.directory / 'metrics/assessment.json')['cells']), 1)

    def lifecycle(self, *, fail_write=False, fail_after_point=False, collection_delay=0, expired=False, unconfirmed=False):
        prepared = self.root / 'prepared'
        workload(prepared, self.model)
        self.directory = self.root / 'inference-serving/lifecycle'
        self.workloads = self.directory / 'raw/inputs/workload'
        self.config['paths']['workspace'] = str(self.root / 'jobs')
        self.workspace = self.root / 'jobs/fixture-campaign' / self.job['name']
        if not fail_after_point:
            self.job['cells'] = self.job['cells'][:1]
        reference = self.root / 'reference.json'; dump(reference, {'fixture': True})
        historical = self.root / 'historical-preflight.json'; dump(historical, {'fixture': True})
        historical_sha = manifest.sha256_file(historical)
        self.config['reference']['historical_preflight_sha256'] = historical_sha
        config_path = self.root / 'config.json'; dump(config_path, self.config)
        packages = {'vllm': '0.24', 'torch': '2.13', 'ray': '2.48'}
        rank = self.ranks[0]
        model_path = runner.runtime.model_path(self.model)
        rank.update(stage='load_model', model_path=model_path, tokenizer_path=model_path,
                    effective_tp=1, effective_pp=1, dtype='torch.bfloat16', versions=packages)
        probe = {'node_id': 'pegasus', 'gpu': {'stdout': 'GPU-fixture, NVIDIA fixture'},
                 'container_probe': {'stdout': json.dumps({'packages': packages})}}
        def remote(node, command, **kwargs):
            if '--phase' in command:
                if fail_after_point and command[command.index('--rate') + 1] == '0.5':
                    raise OSError('later point failed')
                return self.remote(node, command, **kwargs)
            return {'returncode': 0, 'timed_out': False, 'stdout': '', 'stderr': ''}
        original = inf.write_new
        def writer(path, value):
            if fail_write and str(path).endswith('warmup-window.json'):
                raise OSError('disk full')
            original(path, value)
        real_start = manifest.start_run
        real_final = manifest.finalize_run
        def collect(*args):
            local = self.workspace / 'metrics'
            if local.exists():
                shutil.copytree(local, self.directory / 'raw/nodes/pegasus/metrics', dirs_exist_ok=True)
            self.telemetry()
            self.clock += collection_delay
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(manifest, 'capture_git', return_value={'commit': 'a'*40, 'dirty': False}))
            stack.enter_context(mock.patch.object(manifest, 'capture_system', return_value={'hostname': self.node['expected_hostname']}))
            stack.enter_context(mock.patch.object(manifest, 'capture_packages', return_value={}))
            stack.enter_context(mock.patch.object(manifest, 'capture_nvidia', return_value={}))
            stack.enter_context(mock.patch.object(manifest, 'start_run', side_effect=lambda **kw: real_start(**kw, run_id='lifecycle')))
            final = stack.enter_context(mock.patch.object(manifest, 'finalize_run', side_effect=real_final))
            stack.enter_context(mock.patch.object(protocol, 'reference_index_path', return_value=reference))
            stack.enter_context(mock.patch.object(protocol, 'historical_preflight_path', return_value=historical))
            stack.enter_context(mock.patch.object(protocol, 'HISTORICAL_PREFLIGHT_SHA256', historical_sha))
            stack.enter_context(mock.patch.object(protocol, 'verify_hardware', return_value={}))
            stack.enter_context(mock.patch.object(protocol, 'validate_reference_index'))
            stack.enter_context(mock.patch.object(inf, 'stage'))
            stack.enter_context(mock.patch.object(inf, 'wait_ray_head'))
            stack.enter_context(mock.patch.object(inf, 'wait_ready'))
            stack.enter_context(mock.patch.object(inf, '_rank_records', return_value=[rank]))
            tokenize = stack.enter_context(mock.patch.object(inf, '_http',
                side_effect=lambda port, path, request, **kwargs: {'tokens': [int(request['prompt'].split('-')[0])]}))
            stack.enter_context(mock.patch.object(inf, 'collect', side_effect=collect))
            cleanup = stack.enter_context(mock.patch.object(inf, 'cleanup_container',
                side_effect=lambda node, name, directory: {'name': name,
                    'status': 'failed' if unconfirmed else 'confirmed_absent'}))
            stack.enter_context(mock.patch.object(runner.runtime, 'remote', side_effect=remote))
            stack.enter_context(mock.patch.object(inf, 'write_new', side_effect=writer))
            result = inf.run_job(self.config, self.cluster, config_path, self.cluster_path,
                self.job, 'fixture-campaign', {'nodes': [probe]}, prepared, 100000, self.root,
                occupation_deadline=99 if expired else 100000)
        return result, cleanup, final, tokenize


class ReservationTests(unittest.TestCase):
    def test_health_endpoint_and_metrics_accesses_are_bounded_without_real_io(self):
        config, _, _ = protocol.load_config()
        model = config['models'][0]
        deadline = runner.time.monotonic() + 30
        connection = mock.MagicMock()
        response = mock.MagicMock(status=200)
        response.__enter__.return_value = response
        with (mock.patch.object(runner.socket, 'create_connection') as create,
              mock.patch.object(runner.http.client, 'HTTPResponse', return_value=response),
              mock.patch.object(inf, '_http', return_value={'data': [{'id': model['key']}]}) as models,
              mock.patch.object(phases, 'capture', return_value={}) as capture):
            create.return_value.__enter__.return_value = connection
            snapshot = runner._health(config, model, deadline)
        self.assertEqual(snapshot['health_http_status'], 200)
        self.assertIn(b'GET /health HTTP/1.1', connection.sendall.call_args.args[0])
        self.assertLessEqual(create.call_args.kwargs['timeout'], 5)
        self.assertLessEqual(models.call_args.kwargs['timeout'], 5)
        self.assertLessEqual(capture.call_args.args[1], deadline)

    def test_exclusion_release_after_error_and_persistent_inode(self):
        _, cluster_config, _ = protocol.load_config()
        host = inf.cluster.selected_nodes(cluster_config, 'single')[0]['expected_hostname']
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runner.socket, 'gethostname', return_value=host):
            path = Path(temporary) / 'reservation'
            with self.assertRaisesRegex(OSError, 'fixture failure'):
                with runner.reservation(cluster_config, path):
                    inode = path.stat().st_ino
                    with self.assertRaisesRegex(inf.InferenceError, 'activa'):
                        with runner.reservation(cluster_config, path):
                            self.fail('concurrent lock')
                    raise OSError('fixture failure')
            with runner.reservation(cluster_config, path):
                self.assertEqual(path.stat().st_ino, inode)

    def test_foreign_host_rejected_before_lock_creation(self):
        _, cluster_config, _ = protocol.load_config()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runner.socket, 'gethostname', return_value='other'):
            path = Path(temporary) / 'reservation'
            with self.assertRaises(inf.InferenceError):
                with runner.reservation(cluster_config, path):
                    self.fail('wrong host')
            self.assertFalse(path.exists())


class CampaignTests(unittest.TestCase):
    def test_sources_include_new_protocol_and_preserve_historical_contract(self):
        files = runner_sources(protocol.PROTOCOL)
        for name in ('runner', 'protocol', 'client', 'metrics', 'report'):
            self.assertIn(f'infraestructura/serving_capacity_{name}.py', files)
        self.assertEqual(runner_sources('inference-4-2-1-v1'), RUNNER_SOURCES)
        self.assertEqual(len(files), len(set(files)))

    def test_single_campaign_three_blocks_sealed_and_hashed(self):
        self.campaign()

    def test_failed_job_seals_partial_and_keeps_pending_jobs(self):
        self.campaign(failed=True)

    def test_unverified_complete_cannot_seal_complete(self):
        self.campaign(unverified=True)

    def test_open_attempt_is_indexed_and_excluded_from_pending(self):
        self.campaign(mode='open')

    def test_failure_before_manifest_is_not_started_attempt(self):
        self.campaign(mode='not_started')

    def test_sealed_attempt_is_recovered_after_lost_return(self):
        self.campaign(mode='after_seal')

    def test_unconfirmed_retirement_forces_partial(self):
        self.campaign(mode='unconfirmed')

    def test_occupation_limit_forces_partial(self):
        self.campaign(mode='occupation_limit')

    def test_hardware_mismatch_blocks_all_launches(self):
        self.campaign(mode='hardware')

    def test_cpu_closure_overrun_does_not_claim_gpu_overrun(self):
        self.campaign(mode='session_only')

    def test_coordinator_rejects_occupation_out_of_job_order(self):
        self.campaign(mode='out_of_order')

    def test_interrupted_seal_verification_preserves_previous_attempt(self):
        self.campaign(mode='verification_pending')

    def test_interrupted_failed_seal_verification_is_not_open(self):
        self.campaign(mode='verification_pending_failed')

    def test_temporary_preflight_dump_failure_leaves_readable_partial(self):
        self.campaign(mode='preflight_write')

    def campaign(self, *, failed=False, unverified=False, mode=None):
        config, cluster_config, cluster_path = protocol.load_config()
        planned = list(protocol.jobs(config))
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            clock = [100.0]
            stack.enter_context(mock.patch.object(runner.time, 'monotonic', side_effect=lambda: clock[0]))
            ref = root / 'reference.json'; dump(ref, {'fixture': True})
            historical = root / 'historical-preflight.json'; dump(historical, {'fixture': True})
            historical_sha = manifest.sha256_file(historical)
            config['reference']['historical_preflight_sha256'] = historical_sha
            config_path = root / 'protocol.json'; dump(config_path, config)
            work = root / 'workload'; work.mkdir()
            for model in config['models']:
                workload(work, model)
            entered = []
            cut_path = [None]
            real_artifacts = manifest.verify_artifacts
            def artifacts(directory, run=None):
                if Path(directory) == cut_path[0]:
                    raise KeyboardInterrupt('during artifact verification')
                return real_artifacts(directory, run)
            from contextlib import contextmanager
            @contextmanager
            def reserved(*args):
                entered.append(True)
                yield
            def run(*args, **kwargs):
                self.assertTrue(entered)
                job = args[4]
                intent_path = root / 'serving-capacity-campaign' / args[5] / f"attempts/{job['name']}.json"
                intent = inf.read_json(intent_path)
                self.assertEqual(intent['run_id'], kwargs['run_id'])
                if mode == 'not_started':
                    raise KeyboardInterrupt('before manifest')
                model = job['model']
                path = manifest.start_run(kind='inference-serving', model_id=model['id'],
                    model_revision=model['revision'], tokenizer_id=model['tokenizer_id'],
                    tokenizer_revision=model['tokenizer_revision'], requested_config={
                        'protocol': config, 'job': job, 'campaign_id': args[5]},
                    repo_path=root, results_root=root, run_id=kwargs['run_id'])
                if mode in ('verification_pending', 'verification_pending_failed') and job['name'] == planned[1]['name']:
                    cut_path[0] = path
                if mode == 'open':
                    raise KeyboardInterrupt('before seal')
                now = runner.time.monotonic(); utc = manifest.utc_now()
                if mode == 'out_of_order':
                    now -= 10
                occupation = {'started_at': utc, 'started_monotonic_s': now, 'observed_until_at': utc,
                    'observed_until_monotonic_s': now, 'duration_seconds': 0,
                    'removal_confirmed': mode != 'unconfirmed', 'limit_exceeded': mode == 'occupation_limit',
                    'ended_at': utc if mode != 'unconfirmed' else None,
                    'ended_monotonic_s': now if mode != 'unconfirmed' else None}
                dump(path / 'raw/occupation.json', occupation)
                person = {'node_id': 'pegasus', 'rank': 0, 'hostname': 'fixture', 'gpu_uuid': 'GPU-fixture'}
                dump(path / 'raw/rank.json', person)
                manifest.register_participants(path, [{**person, 'evidence_path': 'raw/rank.json'}])
                terminal = 'failed' if failed or mode == 'verification_pending_failed' and cut_path[0] == path else 'complete'
                manifest.finalize_run(path, status=terminal)
                if mode == 'session_only':
                    clock[0] += 1001
                if mode == 'after_seal':
                    raise KeyboardInterrupt('after seal, before index')
                return {'job': job['name'], 'status': 'failed' if failed else 'complete'}
            stack.enter_context(mock.patch.object(protocol, 'frozen_workloads', return_value=work))
            stack.enter_context(mock.patch.object(protocol, 'reference_index_path', return_value=ref))
            stack.enter_context(mock.patch.object(protocol, 'historical_preflight_path', return_value=historical))
            stack.enter_context(mock.patch.object(protocol, 'HISTORICAL_PREFLIGHT_SHA256', historical_sha))
            hardware = stack.enter_context(mock.patch.object(protocol, 'verify_hardware',
                side_effect=ValueError('hardware differs') if mode == 'hardware' else None, return_value={}))
            stack.enter_context(mock.patch.object(protocol, 'validate_reference_index'))
            stack.enter_context(mock.patch.object(manifest, 'capture_git', return_value={'commit': 'a'*40, 'dirty': False}))
            stack.enter_context(mock.patch.object(manifest, 'capture_system', return_value={}))
            stack.enter_context(mock.patch.object(manifest, 'capture_packages', return_value={}))
            stack.enter_context(mock.patch.object(manifest, 'capture_nvidia', return_value={}))
            stack.enter_context(mock.patch.object(manifest, 'verify_artifacts', side_effect=artifacts))
            stack.enter_context(mock.patch.object(runner, 'reservation', side_effect=reserved))
            stack.enter_context(mock.patch.object(runner.runtime, 'preflight', return_value={'errors': [], 'nodes': []}))
            calls = stack.enter_context(mock.patch.object(inf, 'run_job', side_effect=run))
            verify = stack.enter_context(mock.patch.object(runner, '_verified_record',
                side_effect=ValueError('missing archive') if unverified else lambda record, *args: record))
            if mode == 'preflight_write':
                real_dump = inf.json.dump
                interrupted = []
                def preflight_dump(data, stream, **kwargs):
                    if data == {'errors': [], 'nodes': []} and not interrupted:
                        interrupted.append(True)
                        stream.write('{"partial":')
                        raise KeyboardInterrupt('temporary preflight write failure')
                    return real_dump(data, stream, **kwargs)
                stack.enter_context(mock.patch.object(inf.json, 'dump', side_effect=preflight_dump))
            directory = runner.run_campaign(config_path, results_root=root, session_limit_seconds=1000)
            campaign = inf.read_json(directory / 'campaign.json')
            verification_pending = mode in ('verification_pending', 'verification_pending_failed')
            count = 0 if mode in ('hardware', 'preflight_write') else 2 if verification_pending else 1 if failed or unverified or mode else 18
            self.assertEqual(calls.call_count, count)
            self.assertEqual(verify.call_count, 0 if mode in ('hardware', 'preflight_write', 'open', 'not_started') else 1 if verification_pending else count)
            partial = failed or unverified or mode is not None
            self.assertEqual(campaign['status'], 'partial' if partial else 'complete')
            if mode == 'session_only':
                self.assertTrue(campaign['session_limit_exceeded'])
                self.assertFalse(campaign['occupation']['limit_exceeded'])
                self.assertTrue(campaign['occupation']['removal_confirmed'])
            if not partial:
                self.assertFalse(campaign['session_limit_exceeded'])
                self.assertFalse(campaign['occupation']['limit_exceeded'])
            self.assertEqual(campaign['pending_jobs'], [j['name'] for j in planned[calls.call_count:]])
            self.assertEqual(len([key for key in campaign['input_hashes'] if key.startswith('workload/')]), 16)
            self.assertIn('reference/campaign.json', campaign['input_hashes'])
            self.assertIn('reference/preflight.json', campaign['input_hashes'])
            self.assertEqual((directory / 'reference/preflight.json').read_bytes(), historical.read_bytes())
            self.assertEqual((directory / 'cluster.json').read_bytes(), Path(cluster_path).read_bytes())
            marker = inf.read_json(directory / ('PARTIAL' if partial else 'COMPLETE'))
            for name in ('campaign', 'plan', 'preflight'):
                self.assertEqual(marker[name + '_sha256'], manifest.sha256_file(directory / (name + '.json')))
            self.assertEqual(len(campaign['attempts']), count)
            if count:
                attempt = campaign['attempts'][-1]
                self.assertEqual(attempt['status'], 'verification_pending' if verification_pending else mode if mode in ('open', 'not_started') else 'sealed')
                self.assertEqual(attempt['intent_sha256'], manifest.sha256_file(directory / attempt['intent_path']))
                self.assertEqual(len(campaign['runs']), 0 if mode in ('open', 'not_started') else count - 1 if verification_pending else count)
                if verification_pending:
                    self.assertEqual(campaign['attempts'][0]['status'], 'sealed')
                    self.assertEqual(attempt['terminal_status'], 'failed' if mode == 'verification_pending_failed' else 'complete')
                    self.assertNotIn('manifest_sha256', attempt)
            self.assertEqual(hardware.call_count, 0 if mode == 'preflight_write' else 1)
            if calls.call_count == 18:
                self.assertEqual([call.args[4]['block'] for call in calls.call_args_list],
                                 [block for block in (1, 2, 3) for _ in range(6)])
                self.assertEqual(len({call.args[5] for call in calls.call_args_list}), 1)


class IndexProjectionTests(unittest.TestCase):
    def test_second_verified_seal_interrupted_before_index_keeps_seven_points(self):
        self.interrupted_projection('before_index')

    def test_signal_after_reference_keeps_sealed_state_consistent(self):
        self.interrupted_projection('after_index')

    def test_state_published_before_signal_is_recovered_with_hash(self):
        self.interrupted_projection('after_state_publication')

    def interrupted_projection(self, cut):
        # Reuse the reader's real seven-point archive, not a mocked report.
        from tests import test_serving_capacity_report as fixtures
        from infraestructura import serving_capacity_report as report
        fixture = fixtures.ServingCapacityReportTests('test_partial_keeps_audited_point_and_writes_exclusively')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        windows = []; start = 3
        for rate in fixture.config['capacity']['rates_rps']:
            warm_end = start + math.ceil(16 / rate)
            measured_start = warm_end + 6
            measured_end = measured_start + math.ceil(128 / rate)
            windows.append(((start, warm_end), (measured_start, measured_end)))
            start = measured_end + 6
        first = fixture.run_fixture(complete=True, windows=windows)
        first_run = manifest.load_run(first['run_dir'])
        inputs = {value['name']: Path(value['path']) for value in first_run['inputs']}
        first_end = (datetime.fromisoformat(first_run['ended_at']) - fixtures.START).total_seconds()
        second_start = first_end + 1
        second_job = fixture.jobs[1]
        second = fixture.start_fixture_run(second_job, {}, 'pending-run', started=second_start)
        fixture.write_occupation(second, second_start, second_start + 1)
        with mock.patch.object(manifest, 'utc_now', return_value=fixtures.stamp(second_start + 2)):
            manifest.finalize_run(second, status='failed', errors=['fixture interrupted'])
        second_sha = manifest.sha256_file(second / 'manifest.json')
        clock = [fixtures.CLOCK_OFFSET - 2]
        original_index = runner._index_record
        original_verify = runner._verified_record
        original_write = inf.write_new
        interrupted = []
        def index(records, record):
            if cut == 'before_index' and record['job'] == second_job['name']:
                interrupted.append(True)
                raise KeyboardInterrupt('after seal verification, before reference')
            return original_index(records, record)
        def verify(record, *args):
            if cut == 'after_index' and record['job'] == second_job['name']:
                interrupted.append(True)
                raise KeyboardInterrupt('after canonical reference assignment')
            return original_verify(record, *args)
        def write(path, payload):
            original_write(path, payload)
            if (cut == 'after_state_publication' and Path(path).name == second_job['name'] + '-state.json'
                    and not interrupted):
                interrupted.append(True)
                raise KeyboardInterrupt('after state publication, before state hash')
        def run(*args, **kwargs):
            if args[4]['name'] == fixture.jobs[0]['name']:
                self.assertEqual(kwargs['run_id'], 'fixture')
                clock[0] = fixtures.CLOCK_OFFSET + first_end
            else:
                self.assertEqual(kwargs['run_id'], 'pending-run')
                clock[0] = fixtures.CLOCK_OFFSET + second_start + 2
        def preflight(*args, **kwargs):
            clock[0] = fixtures.CLOCK_OFFSET - 1
            return fixture.preflight
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner.time, 'monotonic', side_effect=lambda: clock[0]))
            stack.enter_context(mock.patch.object(manifest, 'utc_now', side_effect=lambda: fixtures.stamp(clock[0] - fixtures.CLOCK_OFFSET)))
            stack.enter_context(mock.patch.object(manifest, 'capture_git', return_value=fixture.git))
            stack.enter_context(mock.patch.object(manifest, 'capture_nvidia', side_effect=AssertionError('GPU capture')))
            stack.enter_context(mock.patch.object(manifest, 'generate_run_id', side_effect=['id', 'fixture', 'pending-run']))
            stack.enter_context(mock.patch.object(protocol, 'load_config', return_value=(fixture.config, fixture.inventory, inputs['cluster'])))
            stack.enter_context(mock.patch.object(protocol, 'make_plan', return_value=fixture.plan))
            stack.enter_context(mock.patch.object(protocol, 'frozen_workloads', return_value=Path(first['run_dir']) / 'raw/inputs/workload'))
            stack.enter_context(mock.patch.object(protocol, 'reference_index_path', return_value=inputs['reference/campaign.json']))
            stack.enter_context(mock.patch.object(protocol, 'historical_preflight_path', return_value=inputs['reference/preflight.json']))
            stack.enter_context(mock.patch.object(inf, '_experiment_inputs', return_value={key: path for key, path in inputs.items()
                if key not in ('cluster', 'protocol', 'reference/campaign.json', 'reference/preflight.json')}))
            stack.enter_context(mock.patch.object(runner, 'reservation'))
            stack.enter_context(mock.patch.object(runner.runtime, 'preflight', side_effect=preflight))
            stack.enter_context(mock.patch.object(runner.runtime, 'remote', side_effect=AssertionError('real service')))
            stack.enter_context(mock.patch.object(inf, 'run_job', side_effect=run))
            stack.enter_context(mock.patch.object(runner, '_index_record', side_effect=index))
            stack.enter_context(mock.patch.object(runner, '_verified_record', side_effect=verify))
            stack.enter_context(mock.patch.object(inf, 'write_new', side_effect=write))
            directory = runner.run_campaign(inputs['protocol'], results_root=fixture.root / 'raw')
        self.assertEqual(interrupted, [True])
        campaign = inf.read_json(directory / 'campaign.json')
        state = campaign['attempts'][1]
        self.assertEqual(campaign['status'], 'partial')
        self.assertEqual(state['status'], 'verification_pending' if cut == 'before_index' else 'sealed')
        self.assertEqual(state['terminal_status'], 'failed')
        self.assertEqual(state['manifest_sha256'], second_sha)
        self.assertEqual(inf.read_json(directory / state['state_path']),
                         {key: value for key, value in state.items() if key not in ('state_path', 'state_sha256')})
        self.assertEqual(state['state_sha256'], manifest.sha256_file(directory / state['state_path']))
        data = report.build_report(directory)
        self.assertEqual(len(data['cells']), 7)
        self.assertEqual({cell['group'] for cell in data['cells']}, {'four'})
        self.assertEqual(len(data['pending_jobs']), 16)
        self.assertEqual(len(campaign['runs']), 1 if cut == 'before_index' else 2)
        self.assertEqual(manifest.sha256_file(Path(first['run_dir']) / 'manifest.json'), first['manifest_sha256'])
        self.assertEqual(manifest.sha256_file(second / 'manifest.json'), second_sha)


if __name__ == '__main__':
    unittest.main()
