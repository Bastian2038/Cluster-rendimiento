"""Bounded serving SLO orchestration on the existing inference run lifecycle.

Importing this module and preparing a plan have no operational effects. Only
``run_campaign`` acquires the Pegasus reservation and enters the authorized run
path. Each point archives its original client phases before the next point.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import http.client
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time

from infraestructura import inference as inf, inference_runtime as runtime, manifest
from infraestructura.inference_evidence import SERVING_CAPACITY_PROTOCOL
from infraestructura.rutas import REPO_ROOT


PROTOCOL = SERVING_CAPACITY_PROTOCOL
LOCK_PATH = Path('/tmp/ipre-serving-capacity.lock')
_COMPILATION = re.compile(r'compil(?:ing|ation)|captur(?:ing|e).*cuda.*graph', re.I)


@contextmanager
def reservation(cluster_config, path=None):
    """Process-owned flock, never unlinked or stolen by a stale PID heuristic."""
    head = inf.cluster.selected_nodes(cluster_config, 'single')[0]
    if head['node_id'] != 'pegasus' or not head['local'] or socket.gethostname() != head['expected_hostname']:
        raise inf.InferenceError('la reserva requiere el host local Pegasus verificado')
    lock = Path(path) if path is not None else LOCK_PATH
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
            raise inf.InferenceError('archivo de reserva local inseguro')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise inf.InferenceError('Pegasus tiene una reserva de serving activa') from exc
        yield
    finally:
        # close releases the flock even after a failed write or interrupt. Keep
        # its inode: unlinking permits two concurrent reservations to succeed.
        os.close(fd)


def _expected(config, model, phase, workload_directory):
    from infraestructura.serving_capacity_client import expected_prompts
    rows = expected_prompts(Path(workload_directory) / model['key'], phase,
                            config['measurement'][phase + '_requests'])
    return [row['prompt_sha256'] for row in rows]


def _summarize(receipt, config, model, rate, phase, workload_directory):
    from infraestructura.serving_capacity_metrics import summarize_phase
    summary = summarize_phase(receipt, config, model['key'], rate, phase,
                               _expected(config, model, phase, workload_directory))
    if not isinstance(summary, dict) or summary.get('valid') is False:
        raise inf.InferenceError(f'{phase}: fase del cliente inválida: {summary}')
    return summary


def _snapshot_identity(snapshot, config, model):
    from infraestructura import inference_diag_phases as phases
    if (snapshot.get('valid') is not True or snapshot.get('drained') is not True
            or snapshot.get('http_status') != 200 or snapshot.get('model_key') != model['key']
            or snapshot.get('port') != config['runtime']['api_port']):
        raise inf.InferenceError('servidor no saludable o no drenado')
    running, waiting = phases._gauges(*phases._parse(snapshot['body'])[:2])
    if not phases._model_series(running, model['key']) or any(running.values()) or any(waiting.values()):
        raise inf.InferenceError('métricas originales no acreditan identidad y drenaje')
    # Independent continuity of the actual loopback listener, even for failed
    # requests that cannot be validated with successful-request histograms.
    return phases._receipt(snapshot)


def _health(config, model, deadline):
    from infraestructura import inference_diag_phases as phases
    limit = min(deadline, time.monotonic() + 30)
    port = config['runtime']['api_port']
    health_deadline = min(limit, time.monotonic() + 5)
    with socket.create_connection(('127.0.0.1', port), timeout=inf.remaining(health_deadline, 5)) as connection:
        connection.settimeout(inf.remaining(health_deadline, 5))
        connection.sendall(f'GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n'.encode('ascii'))
        with http.client.HTTPResponse(phases._DeadlineReader(connection, health_deadline)) as response:
            response.begin()
            health_status = response.status
    models = inf._http(config['runtime']['api_port'], '/v1/models',
                       timeout=inf.remaining(limit, 5))
    snapshot = phases.capture(config['runtime']['api_port'], limit, model['key'])
    snapshot['models'] = models
    snapshot['health_http_status'] = health_status
    return snapshot


def _check_health(snapshot, config, model):
    if snapshot.get('health_http_status') != 200:
        raise inf.InferenceError('el endpoint /health no acredita salud del motor')
    if [row['id'] for row in snapshot.get('models', {}).get('data', [])] != [model['key']]:
        raise inf.InferenceError('identidad del servidor cambió durante el punto')
    return _snapshot_identity(snapshot, config, model)


def _phase_evidence(directory, config, job, rate, phase):
    from infraestructura.serving_capacity_protocol import point_name
    name = point_name(rate)
    model = job['model']
    raw = inf.read_json(directory / 'raw/nodes/pegasus/metrics' / name / (phase + '.json'))
    summary = _summarize(raw, config, model, rate, phase, directory / 'raw/inputs/workload')
    dest = directory / 'raw/serving-capacity' / name
    window = inf.read_json(dest / (phase + '-window.json'))
    if (window.get('client_returncode') != 0 or window.get('client_timed_out') is not False
            or window.get('cell') != name or window.get('phase') != phase):
        raise inf.InferenceError('cliente sin terminación verificada para esta fase')
    start = datetime.fromisoformat(window['measurement_started_at'])
    end = datetime.fromisoformat(window['measurement_ended_at'])
    if start.utcoffset() is None or end.utcoffset() is None or end <= start:
        raise inf.InferenceError('ventana de fase inválida')
    span = window['ended_monotonic_s'] - window['started_monotonic_s']
    if min(span, (end - start).total_seconds()) < summary['last_terminal_s']:
        raise inf.InferenceError('ventana externa no cubre la última respuesta del cliente')
    before, after = (inf.read_json(dest / (phase + '-' + side + '.json')) for side in ('before', 'after'))
    if _check_health(before, config, model) != _check_health(after, config, model):
        raise inf.InferenceError('proceso del servidor cambió durante la fase')
    if not before['finished_monotonic_s'] <= window['started_monotonic_s'] < window['ended_monotonic_s'] <= after['started_monotonic_s']:
        raise inf.InferenceError('ventana del cliente fuera del control de salud')
    if inf.read_json(dest / (phase + '-summary.json')) != summary:
        raise inf.InferenceError('resumen de fase difiere del raw')
    return summary, window, before, after


def require_tokenization(directory, config):
    """Require the sealed server-tokenization receipt for both input profiles."""
    value = inf.read_json(manifest._internal_file(Path(directory), 'raw/tokenization-verified.json'))
    profiles = value.get('profiles')
    if (value.get('verified') is not True or not isinstance(profiles, list)
            or any(type(length) is not int for length in profiles)
            or profiles != config['workload']['input_tokens']):
        raise inf.InferenceError('tokenización del servidor sin verificar')
    return value


def require_point_quality(directory, name):
    """Require a positive point-quality receipt; readers also recompute quality."""
    if not isinstance(name, str) or not inf.SAFE.fullmatch(name):
        raise inf.InferenceError('nombre de punto inválido')
    value = inf.read_json(manifest._internal_file(Path(directory), f'raw/diagnostic/{name}/quality.json'))
    if value.get('valid') is not True or value.get('issues') != []:
        raise inf.InferenceError('cierre no acredita calidad por punto')
    return value


def require_occupation(directory):
    """Validate retirement timing without conflating collection with occupation."""
    value = inf.read_json(manifest._internal_file(Path(directory), 'raw/occupation.json'))
    start = value['started_monotonic_s']; observed = value['observed_until_monotonic_s']
    if (type(start) not in (int, float) or type(observed) not in (int, float)
            or not math.isfinite(start) or not math.isfinite(observed)
            or not 0 <= start <= observed or value['duration_seconds'] != observed - start
            or type(value['removal_confirmed']) is not bool or type(value['limit_exceeded']) is not bool):
        raise inf.InferenceError('recibo de ocupación inválido')
    started = datetime.fromisoformat(value['started_at'])
    observed_at = datetime.fromisoformat(value['observed_until_at'])
    if started.utcoffset() is None or observed_at.utcoffset() is None or observed_at < started:
        raise inf.InferenceError('marcas UTC de ocupación inválidas')
    if value['removal_confirmed']:
        if value['ended_monotonic_s'] != observed or value['ended_at'] != value['observed_until_at']:
            raise inf.InferenceError('retirada confirmada sin marca de fin coherente')
    elif value.get('ended_at') is not None or value.get('ended_monotonic_s') is not None:
        raise inf.InferenceError('ocupación abierta no admite fin confirmado')
    return value


def run_point(config, job, cell, nodes, probes, workspace, campaign_id, directory,
              deadline, active_clients, ranks):
    """Run two CPU clients; persist raw and drained receipts before advancing."""
    from infraestructura import serving_capacity_protocol as protocol
    rate = cell['rate_rps']; name = protocol.point_name(rate); model = job['model']
    dest = directory / 'raw/serving-capacity' / name
    previous = None
    index = job['cells'].index(cell)
    if index:
        last = protocol.point_name(job['cells'][index - 1]['rate_rps'])
        prior = inf.read_json(directory / 'raw/serving-capacity' / last / 'measured-after.json')
        previous = _check_health(prior, config, model)
    for phase in ('warmup', 'measured'):
        before = _health(config, model, deadline)
        inf.write_new(dest / (phase + '-before.json'), before)
        identity = _check_health(before, config, model)
        if previous is not None and previous != identity:
            raise inf.InferenceError('servidor cambió entre fases o puntos')
        offsets = inf.execution_logs(nodes, workspace, deadline, ranks) if phase == 'measured' else None
        left = inf.remaining(deadline, config['measurement']['cell_limit_seconds'])
        container = f"ipre-{campaign_id[-8:]}-{job['name']}-{name}-{phase}"
        command = inf.container_command(nodes[0], config, probes['pegasus'], workspace, container,
            protocol.client_command(config, model, rate, phase), gpu=False, seconds=left - 2)
        command.insert(2, '--rm')
        # Registered before any command or receipt write; the enclosing
        # run_job finally always cleans it up after an exception or ENOSPC.
        active_clients.append((nodes[0], container))
        start = manifest.utc_now(); started = time.monotonic()
        result = None; execution_error = None
        try:
            result = runtime.remote(nodes[0], command, timeout=inf.remaining(deadline, left))
        except (Exception, KeyboardInterrupt) as exc:
            execution_error = exc
        ended = time.monotonic(); end = manifest.utc_now()
        if result is None:
            timeout = isinstance(execution_error, (TimeoutError, subprocess.TimeoutExpired))
            result = {'returncode': None, 'timed_out': True if timeout else None, 'stdout': '',
                      'stderr': f'{type(execution_error).__name__}: {execution_error}',
                      'error_type': type(execution_error).__name__, 'error_reason': str(execution_error)}
        window = {'cell': name, 'phase': phase,
                  'measurement_started_at': start, 'measurement_ended_at': end,
                  'started_monotonic_s': started, 'ended_monotonic_s': ended,
                  'client_returncode': result.get('returncode'),
                  'client_timed_out': result.get('timed_out', False)}
        if execution_error is not None:
            window.update(client_error_type=type(execution_error).__name__,
                          client_error_reason=str(execution_error))
        inf.write_new(dest / (phase + '-window.json'), window)
        inf.write_new(directory / 'logs' / f'{name}-{phase}.json', result)
        if execution_error is not None:
            raise execution_error
        raw = inf._archive_client_file(directory, workspace, name, phase + '.json')
        runtime.checked(result, f'{name} {phase}')
        active_clients.remove((nodes[0], container))
        summary = _summarize(raw, config, model, rate, phase, directory / 'raw/inputs/workload')
        inf.write_new(dest / (phase + '-summary.json'), summary)
        after = _health(config, model, deadline)
        inf.write_new(dest / (phase + '-after.json'), after)
        previous = _check_health(after, config, model)
        if identity != previous:
            raise inf.InferenceError('servidor cambió durante el cliente')
        if phase == 'measured':
            window_logs = inf.execution_logs(nodes, workspace, deadline, ranks, offsets)
            receipt = {'cell': name, 'measurement_started_at': start, 'measurement_ended_at': end,
                       'logs': {node: {filename: value['bounds'] for filename, value in files.items()}
                                for node, files in window_logs.items()}}
            inf.write_new(directory / 'raw/diagnostic' / name / 'execution-window.json', receipt)
            if any(_COMPILATION.search(value['text']) for files in window_logs.values() for value in files.values()):
                raise inf.InferenceError('compilación observada durante la medida')
            measured_window = window
    # Re-read both archived phases and their summaries. No later client's
    # filesystem contents can replace these originals during final collection.
    for phase in ('warmup', 'measured'):
        _phase_evidence(directory, config, job, rate, phase)
    record = {**summary, **cell, 'cell': name, 'status': 'valid',
              'measurement_started_at': measured_window['measurement_started_at'],
              'measurement_ended_at': measured_window['measurement_ended_at']}
    inf.write_new(directory / 'metrics' / (name + '.json'), record)
    return record


def point_issues(directory, config, job, cell, ranks):
    """Recompute quality independently of job success, using archived files."""
    from infraestructura import inference_diag_quality as quality
    issues = []
    try:
        phases = [_phase_evidence(directory, config, job, cell['rate_rps'], phase)
                  for phase in ('warmup', 'measured')]
        if _check_health(phases[0][3], config, job['model']) != _check_health(phases[1][2], config, job['model']):
            raise inf.InferenceError('identidad cambió entre fases')
        if phases[0][3]['finished_monotonic_s'] > phases[1][2]['started_monotonic_s']:
            raise inf.InferenceError('fases solapadas o invertidas')
        window = phases[1][1]
        if any(cell.get(key) != window[key] for key in ('measurement_started_at', 'measurement_ended_at')):
            raise inf.InferenceError('intervalo medido difiere del recibo original')
        if any(cell.get(key) != value for key, value in phases[1][0].items() if key != 'status'):
            raise inf.InferenceError('celda difiere del resumen medido')
        issues.extend(quality.cell_issues(directory, cell, ranks))
        # Warmup also requires healthy telemetry and drainage. Compilation is
        # expected in warmup, so apply the log compilation rule to measurement.
        warm = phases[0][1]
        for rank in ranks:
            quality.telemetry_window(directory / 'raw/nodes' / rank['node_id'] / 'logs/telemetry.jsonl',
                datetime.fromisoformat(warm['measurement_started_at']),
                datetime.fromisoformat(warm['measurement_ended_at']), rank)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        issues.append(f'point evidence: {exc}')
    return issues


def assess_job(directory, config, job, ranks):
    """Salvage only complete points whose own evidence passes final quality."""
    from infraestructura.serving_capacity_protocol import point_name
    cells = []; errors = []
    for coordinate in job['cells']:
        name = point_name(coordinate['rate_rps'])
        path = directory / 'metrics' / (name + '.json')
        if not path.is_file():
            continue
        try:
            cell = inf.read_json(path)
            issues = point_issues(directory, config, job, cell, ranks)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            cell = {**coordinate, 'cell': name}
            issues = [f'point record: {exc}']
        if cell.get('cell') != name or cell.get('rate_rps') != coordinate['rate_rps']:
            issues.append('point coordinate differs from planned cell')
        cell['status'] = 'invalid' if issues else 'valid'
        inf.write_new(directory / 'raw/diagnostic' / name / 'quality.json',
                      {'valid': not issues, 'issues': issues})
        if not issues:
            require_point_quality(directory, name)
        cells.append(cell)
        errors.extend(f'{name}: {issue}' for issue in issues)
    return cells, errors


def _verified_record(record, job, config, campaign_id, input_hashes):
    """Reject a purported COMPLETE lacking its sealed point or input evidence."""
    directory = Path(record['run_dir'])
    if record.get('manifest_sha256') != manifest.sha256_file(directory / 'manifest.json'):
        raise inf.InferenceError('hash de manifiesto de subrun difiere')
    run = manifest.load_run(directory)
    manifest.verify_artifacts(directory, run)
    if (record.get('status') != run['status'] or record.get('job') != job['name']
            or record.get('group') != job['group'] or record.get('block') != job['block']
            or record.get('model_key') != job['model']['key']):
        raise inf.InferenceError('referencia de subrun contradice su identidad o cierre')
    request = run['requested_config']
    if request != {'protocol': config, 'job': job, 'campaign_id': campaign_id}:
        raise inf.InferenceError('subrun pertenece a otro job o campaña')
    archived_protocol = inf.read_json(directory / 'raw/inputs/protocol')
    if archived_protocol != config:
        raise inf.InferenceError('protocolo archivado difiere del efectivo')
    inputs = {item['name']: item['sha256'] for item in run['inputs']}
    if inputs != input_hashes or len(inputs) != len(run['inputs']):
        raise inf.InferenceError('inputs del subrun difieren de la identidad de campaña')
    for name, digest in input_hashes.items():
        if manifest.sha256_file(directory / 'raw/inputs' / name) != digest:
            raise inf.InferenceError(f'input archivado difiere: {name}')
    assessment = inf.read_json(directory / 'metrics/assessment.json')
    from infraestructura.serving_capacity_protocol import point_name
    planned = {point_name(cell['rate_rps']): cell for cell in job['cells']}
    observed = assessment.get('cells', [])
    if [cell['rate_rps'] for cell in observed] != [cell['rate_rps'] for cell in job['cells'][:len(observed)]]:
        raise inf.InferenceError('orden de puntos difiere del plan')
    if len({cell['cell'] for cell in observed}) != len(observed):
        raise inf.InferenceError('celdas duplicadas en cierre')
    ranks = inf.read_json(directory / 'raw/startup.json')['ranks']
    preflight = inf.read_json(directory / 'raw/preflight.json')
    _, _, comparable = inf._observed_topology(directory, run, job, preflight, config, None)
    observed_ranks = [inf.read_json(directory / row['evidence_path']) for row in run['participants']]
    if not comparable or sorted(observed_ranks, key=lambda r: r['rank']) != sorted(ranks, key=lambda r: r['rank']):
        raise inf.InferenceError('ranks o entorno observado no acreditan la topología')
    require_tokenization(directory, config)
    occupation = require_occupation(directory)
    if (datetime.fromisoformat(occupation['started_at']) < datetime.fromisoformat(run['started_at'])
            or datetime.fromisoformat(occupation['observed_until_at']) > datetime.fromisoformat(run['ended_at'])):
        raise inf.InferenceError('ocupación queda fuera de los límites del subrun sellado')
    valid = []
    for cell in observed:
        if cell['cell'] not in planned or cell.get('rate_rps') != planned[cell['cell']]['rate_rps']:
            raise inf.InferenceError('celda fuera del plan')
        if cell.get('status') != 'valid':
            continue
        issues = point_issues(directory, config, job, cell, ranks)
        require_point_quality(directory, cell['cell'])
        if issues:
            raise inf.InferenceError('cierre no acredita calidad por punto')
        valid.append(cell['cell'])
    complete = (record.get('status') == run['status'] == 'complete' and assessment.get('valid') is True
                and not assessment.get('errors') and len(valid) == len(planned))
    if complete:
        cleanup = inf.read_json(directory / 'raw/cleanup.json')
        if (len(cleanup) < len(run['participants']) or any(item.get('status') != 'confirmed_absent'
                or item.get('errors') for item in cleanup) or run.get('errors') or record.get('errors')
                or not occupation['removal_confirmed'] or occupation['limit_exceeded']):
            raise inf.InferenceError('COMPLETE sin limpieza confirmada o con errores')
    if record.get('status') == 'complete' and not complete:
        raise inf.InferenceError('COMPLETE sin evidencia íntegra de todos los puntos')
    return {**record, 'valid_cells': valid, 'occupation': occupation}


def recover_run_reference(attempt, job, config, campaign_id):
    """Rebuild an index reference from a verified closed manifest, without writes.

    This proves the attempt identity and archive seal. Point eligibility is
    separately rechecked by ``_verified_record`` and by the report reader.
    """
    directory = Path(attempt['run_dir'])
    run = manifest.load_run(directory)
    manifest.verify_artifacts(directory, run)
    expected_model = job['model']
    if (attempt['job'] != job['name'] or run['run_id'] != attempt['run_id']
            or directory.name != attempt['run_id'] or run['kind'] != 'inference-serving'
            or run['status'] not in ('complete', 'failed')
            or run['requested_config'] != {'protocol': config, 'job': job, 'campaign_id': campaign_id}
            or run['model'] != {'id': expected_model['id'], 'revision': expected_model['revision'],
                'tokenizer': {'id': expected_model['tokenizer_id'], 'revision': expected_model['tokenizer_revision']}}
            or run['policy']['strict'] is not True or run['git']['dirty'] is not False):
        raise inf.InferenceError('manifiesto cerrado no corresponde al intento reservado')
    errors = [value.get('message', str(value)) if isinstance(value, dict) else str(value) for value in run['errors']]
    return {'job': job['name'], 'group': job['group'], 'block': job['block'],
            'model_key': expected_model['key'], 'run_dir': str(directory), 'status': run['status'],
            'manifest_sha256': manifest.sha256_file(directory / 'manifest.json'), 'errors': errors}


def _copy_input(source, target, digest):
    contents = Path(source).read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(contents)
    if manifest.sha256_file(target) != digest:
        raise inf.InferenceError(f'copia de entrada cambió: {target.name}')


def _finish_attempt(attempt, job, config, campaign_id):
    directory = Path(attempt['run_dir'])
    if not directory.exists():
        attempt['status'] = 'not_started'
        return None
    markers = [name for name in ('COMPLETE', 'FAILED') if (directory / name).exists()]
    attempt['status'] = 'verification_pending' if markers else 'open'
    if len(markers) == 1:
        attempt['terminal_status'] = 'complete' if markers[0] == 'COMPLETE' else 'failed'
    run = manifest.load_run(directory)
    if run['status'] == 'running':
        return None
    attempt.update(status='verification_pending', terminal_status=run['status'])
    record = recover_run_reference(attempt, job, config, campaign_id)
    # Verification alone does not index a reference. Keep the observed seal
    # pending; public status is projected from the canonical reference tuple.
    attempt['manifest_sha256'] = record['manifest_sha256']
    return record


def _index_record(records, record):
    """Prepare the single reference transition; assignment commits it at once."""
    return (*records, record)


def _attempt_projection(attempts, records):
    """Derive both public states and references from one immutable snapshot."""
    refs = tuple(records)
    by_job = {record['job']: record for record in refs}
    projected = []
    for observed in attempts:
        value = {key: item for key, item in observed.items() if key not in ('state_path', 'state_sha256')}
        reference = by_job.get(value['job'])
        if reference is not None:
            value.update(status='sealed', manifest_sha256=reference['manifest_sha256'],
                         terminal_status=reference['status'])
        elif value['status'] == 'sealed':
            value['status'] = 'verification_pending'
        projected.append(value)
    return refs, projected


def _publish_attempt_state(directory, attempt):
    """Publish or recover this exact projection, never replacing a journal."""
    relative = f"attempts/{attempt['job']}-state.json"
    path = directory / relative
    if not os.path.lexists(path):
        inf.write_new(path, attempt)
    if inf.read_json(path) != attempt:
        raise inf.InferenceError('estado inmutable difiere de la proyección de referencias')
    return {**attempt, 'state_path': relative, 'state_sha256': manifest.sha256_file(path)}


def _journal_projection(directory, attempts, records, errors):
    refs, projected = _attempt_projection(attempts, records)
    journal = []
    for attempt in projected:
        try:
            snapshot = _publish_attempt_state(directory, attempt)
        except (Exception, KeyboardInterrupt) as exc:
            errors.append(f"{attempt['job']}: estado de intento: {type(exc).__name__}: {exc}")
            # The atomic publication may have succeeded just before the signal,
            # including before state_path/hash could enter memory. Recover it
            # from the same projection; a closed file is never changed.
            snapshot = _publish_attempt_state(directory, attempt)
        journal.append(snapshot)
    return refs, journal


def run_campaign(config_path, *, results_root=REPO_ROOT / 'resultados/raw', session_limit_seconds=None):
    """Execute one three-block identity; caller must hold specific authorization."""
    from infraestructura import serving_capacity_protocol as protocol
    config_path = Path(config_path).resolve()
    config, cluster_config, cluster_path = protocol.load_config(config_path)
    maximum = config['measurement']['session_limit_seconds']
    limit = maximum if session_limit_seconds is None else session_limit_seconds
    reserve = config['measurement']['cleanup_reserve_seconds']
    if type(limit) is not int or not reserve < limit <= maximum <= 43200:
        raise inf.InferenceError('límite de campaña inválido o sin reserva de cierre')
    if inf.read_json(config_path) != config:
        raise inf.InferenceError('config versionado debe contener el protocolo efectivo completo')
    plan = protocol.make_plan(config_path)
    planned_jobs = list(protocol.jobs(config))
    if plan['protocol'] != config:
        raise inf.InferenceError('plan preparado con otra configuración efectiva')
    workloads = protocol.frozen_workloads(config)
    reference = protocol.reference_index_path(config)
    historical_preflight = protocol.historical_preflight_path(config)
    protocol.validate_reference_index(config, reference)
    inputs = {'cluster': cluster_path, 'protocol': config_path, 'reference/campaign.json': reference,
              'reference/preflight.json': historical_preflight,
              **inf._experiment_inputs(workloads, PROTOCOL)}
    input_hashes = {name: manifest.sha256_file(path) for name, path in inputs.items()}
    git = manifest.capture_git(REPO_ROOT)
    if not git['commit'] or git['dirty']:
        raise inf.InferenceError('campaña requiere commit identificable y árbol limpio')
    with reservation(cluster_config):
        campaign_id = manifest.generate_run_id()
        directory = Path(results_root) / 'serving-capacity-campaign' / campaign_id
        directory.mkdir(parents=True, exist_ok=False)
        inf.write_new(directory / 'plan.json', plan)
        _copy_input(historical_preflight, directory / 'reference/preflight.json', input_hashes['reference/preflight.json'])
        _copy_input(cluster_path, directory / 'cluster.json', input_hashes['cluster'])
        started = time.monotonic(); started_at = manifest.utc_now()
        deadline = started + limit - reserve
        effective_deadline = started + limit
        records = (); errors = []; attempts = []; check = None; hardware = None
        prior_end = started; prior_utc = datetime.fromisoformat(started_at)
        removal_confirmed = True; last_end = started; last_end_at = started_at
        try:
            check = runtime.preflight(cluster_config, config, container_probe=True)
            inf.write_new(directory / 'preflight.json', check)
            if check['errors']:
                raise inf.InferenceError('preflight bloqueado: ' + '; '.join(check['errors']))
            hardware = protocol.verify_hardware(config, directory / 'reference/preflight.json', check)
            for job in planned_jobs:
                inf.remaining(deadline, 1)
                current_git = manifest.capture_git(REPO_ROOT)
                if current_git['commit'] != git['commit'] or current_git['dirty']:
                    raise inf.InferenceError('commit o estado Git cambió durante la campaña')
                # Hash all inputs before each launch, binding every block to
                # this campaign's original code/config/reference/workload set.
                if {name: manifest.sha256_file(path) for name, path in inputs.items()} != input_hashes:
                    raise inf.InferenceError('inputs de campaña cambiaron antes del job')
                run_id = manifest.generate_run_id()
                intent_path = f"attempts/{job['name']}.json"
                intent = {'schema_version': 1, 'campaign_id': campaign_id, 'job': job['name'],
                          'run_id': run_id, 'run_dir': str(Path(results_root).resolve() / 'inference-serving' / run_id),
                          'requested_at': manifest.utc_now(), 'requested_monotonic_s': time.monotonic()}
                inf.write_new(directory / intent_path, intent)
                attempt = {key: intent[key] for key in ('job', 'run_id', 'run_dir')}
                attempt.update(status='not_started', intent_path=intent_path,
                               intent_sha256=manifest.sha256_file(directory / intent_path))
                attempts.append(attempt)
                job_errors = []; record = None
                try:
                    inf.run_job(config, cluster_config, config_path, cluster_path, job, campaign_id,
                        check, workloads, deadline, results_root, run_id=run_id, occupation_deadline=effective_deadline)
                except (Exception, KeyboardInterrupt) as exc:
                    job_errors.append(f'{type(exc).__name__}: {exc}')
                # Inspect the reserved path even if the call failed after it
                # sealed. A returned Python value is never the index authority.
                try:
                    record = _finish_attempt(attempt, job, config, campaign_id)
                    if record is not None:
                        records = _index_record(records, record)
                        verified = _verified_record(record, job, config, campaign_id, input_hashes)
                        records = (*records[:-1], verified)
                    else:
                        job_errors.append(f"subrun {attempt['status']}: sin cierre verificable")
                except (Exception, KeyboardInterrupt) as exc:
                    job_errors.append(f'verificación de intento: {type(exc).__name__}: {exc}')
                occupation = None
                if attempt['status'] != 'not_started':
                    try:
                        occupation = require_occupation(Path(attempt['run_dir']))
                        actual_start = datetime.fromisoformat(occupation['started_at'])
                        if occupation['started_monotonic_s'] < prior_end or actual_start < prior_utc:
                            raise inf.InferenceError('orden de ocupación contradice los jobs consecutivos')
                        if not occupation['removal_confirmed']:
                            raise inf.InferenceError('retirada sin confirmar; ocupación permanece abierta')
                        last_end = occupation['ended_monotonic_s']; last_end_at = occupation['ended_at']
                        prior_end = last_end; prior_utc = datetime.fromisoformat(last_end_at)
                        if occupation['limit_exceeded'] or last_end > effective_deadline:
                            job_errors.append('ocupación excedió el límite efectivo')
                    except (Exception, KeyboardInterrupt) as exc:
                        removal_confirmed = False
                        job_errors.append(f'ocupación: {type(exc).__name__}: {exc}')
                attempt['occupation'] = occupation
                attempt['errors'] = job_errors.copy()
                _, projected = _attempt_projection(attempts, records)
                _publish_attempt_state(directory, projected[-1])
                errors.extend(f"{job['name']}: {error}" for error in job_errors)
                if job_errors or record is None or record['status'] != 'complete':
                    errors.append(f"{job['name']}: sesión detenida tras fallo, sin reintentos")
                    break
        except (Exception, KeyboardInterrupt) as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
        records, indexed_attempts = _journal_projection(directory, attempts, records, errors)
        if not (directory / 'preflight.json').exists():
            inf.write_new(directory / 'preflight.json', check or {'errors': errors, 'nodes': []})
        ended = time.monotonic(); ended_at = manifest.utc_now()
        if not attempts:
            last_end = ended; last_end_at = ended_at
        observed = last_end if removal_confirmed else ended
        observed_at = last_end_at if removal_confirmed else ended_at
        occupation_exceeded = observed > effective_deadline
        session_exceeded = ended > effective_deadline
        if session_exceeded: errors.append('límite efectivo de sesión excedido')
        if occupation_exceeded: errors.append('límite efectivo de ocupación excedido')
        status = 'complete' if len(records) == len(planned_jobs) and not errors else 'partial'
        campaign = {'campaign_id': campaign_id, 'status': status, 'git': git, 'protocol': config,
                    'input_hashes': input_hashes, 'runs': records, 'attempts': indexed_attempts, 'errors': errors,
                    'hardware': hardware,
                    'pending_jobs': [job['name'] for job in planned_jobs[len(attempts):]],
                    'started_at': started_at, 'ended_at': ended_at,
                    'started_monotonic_s': started, 'ended_monotonic_s': ended,
                    'duration_seconds': ended - started,
                    'session_limit_exceeded': session_exceeded,
                    'occupation': {'started_at': started_at, 'started_monotonic_s': started,
                        'ended_at': last_end_at if removal_confirmed else None,
                        'ended_monotonic_s': last_end if removal_confirmed else None,
                        'observed_until_at': observed_at, 'observed_until_monotonic_s': observed,
                        'duration_seconds': observed-started, 'removal_confirmed': removal_confirmed,
                        'limit_exceeded': occupation_exceeded},
                    'effective_session_limit_seconds': limit}
        inf.write_new(directory / 'campaign.json', campaign)
        inf.write_new(directory / ('COMPLETE' if status == 'complete' else 'PARTIAL'),
            {name + '_sha256': manifest.sha256_file(directory / (name + '.json'))
             for name in ('campaign', 'plan', 'preflight')})
        return directory
