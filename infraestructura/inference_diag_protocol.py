"""Pure campaign coordinates and pinned inputs for the approved diagnosis."""

from pathlib import Path

PROTOCOL = 'inference-diagnosis-2048-c8-v1'
COORDINATES = {'A': (2048, 8), 'B': (2048, 4), 'C': (512, 8)}
ROTATIONS = ('ABC', 'BCA', 'CAB')
DIAGNOSTIC_SOURCES = (
    'infraestructura/inference_diag_protocol.py',
    'infraestructura/inference_diag_phases.py',
    'infraestructura/inference_diag_client.py',
    'infraestructura/inference_diag_traces.py',
    'infraestructura/inference_diag_quality.py',
)


def cell_id(cell):
    name = f"{cell['input_tokens']}-c{cell['concurrency']}"
    return name + ('-' + cell['mode'] if 'mode' in cell else '')


def jobs(config):
    for block, rotation in enumerate(ROTATIONS, 1):
        for model in config['models']:
            for group in ('four', 'pair', 'single'):
                yield dict(name=f"b{block}-{model['key']}-{group}", block=block,
                           model=model, group=group, series='primary', cells=[
                               dict(input_tokens=COORDINATES[key][0], concurrency=COORDINATES[key][1])
                               for key in rotation])
    for model in config['models']:
        for group in ('four', 'pair'):
            yield dict(name=f"b4-{model['key']}-{group}", block=4,
                       model=model, group=group, series='auxiliary', cells=[
                           dict(input_tokens=COORDINATES[key][0], concurrency=COORDINATES[key][1], mode=mode)
                           for mode in ('off', 'on') for key in 'ABC'])


def cell_matrix_issues(job, cells, *, complete):
    """Validate actual cell identities without treating an empty matrix as valid."""
    if not isinstance(cells, list):
        return ['cell matrix must be a list']
    expected = {cell_id(c): c for c in job['cells']}
    seen = set(); issues = []
    for cell in cells:
        if not isinstance(cell, dict):
            issues.append('invalid cell record'); continue
        name = cell.get('cell')
        coordinate = expected.get(name)
        if name in seen:
            issues.append(f'{name}: duplicated cell')
        seen.add(name)
        if coordinate is None or any(type(cell.get(k)) is not type(v) or cell[k] != v
                                     for k, v in (coordinate or {}).items()):
            issues.append(f'{name}: coordinates differ from the server plan')
        if coordinate is not None and ('mode' in cell) != ('mode' in coordinate):
            issues.append(f'{name}: unexpected/missing cell mode')
        if complete and cell.get('status') != 'valid':
            issues.append(f'{name}: COMPLETE requires a valid client batch')
    if complete and (not cells or seen != set(expected)):
        issues.append('COMPLETE requires the exact nonempty cell matrix')
    return issues


def cell_counts(series, cells):
    return {'primary': sum(series == 'primary' for c in cells),
            'off': sum(series == 'auxiliary' and c.get('mode') == 'off' for c in cells),
            'on': sum(series == 'auxiliary' and c.get('mode') == 'on' for c in cells)}


def historical_path(config, root):
    path = Path(config['diagnosis']['historical_campaign'])
    return path if path.is_absolute() else Path(root) / path


def validate_config(config):
    """Validate the public pin without needing access to historical raw."""
    import re
    pin = config['diagnosis']
    if not re.fullmatch(r'[0-9a-f]{64}', pin.get('historical_campaign_sha256', '')):
        raise ValueError('historical campaign requires a SHA-256 pin')
    path = Path(pin.get('historical_campaign', ''))
    if '..' in path.parts or path.name != 'campaign.json':
        raise ValueError('invalid historical campaign path')
    expected = {f'workload/{model["key"]}/{length}/{phase}.jsonl'
                for model in config['models'] for length in (512, 2048) for phase in ('warmup', 'measured')}
    hashes = pin.get('historical_jsonl_sha256', {})
    if set(hashes) != expected or any(not re.fullmatch(r'[0-9a-f]{64}', v) for v in hashes.values()):
        raise ValueError('eight immutable historical JSONL pins required')


def validate_archived_historical(config, path):
    import json
    from infraestructura import manifest
    if manifest.sha256_file(path) != config['diagnosis']['historical_campaign_sha256']:
        raise ValueError('archived historical campaign SHA-256 differs from the pin')
    historical = json.loads(Path(path).read_text(encoding='utf-8'))
    if historical['status'] != 'complete' or historical['errors'] or historical['pending_jobs']:
        raise ValueError('archived historical campaign is incomplete')
    for key in ('models', 'image', 'runtime', 'workload', 'network'):
        if config[key] != historical['protocol'][key]:
            raise ValueError(f'{key} differs from the archived historical campaign')
    expected = {k: v for k, v in historical['input_hashes'].items() if k.endswith('.jsonl')}
    if expected != config['diagnosis']['historical_jsonl_sha256']:
        raise ValueError('archived historical JSONL pins differ')


def validate_historical(config, root):
    """Read a pinned local index; never retokenize or alter historical raw."""
    import json
    from infraestructura import manifest
    path = historical_path(config, root)
    if manifest.sha256_file(path) != config['diagnosis']['historical_campaign_sha256']:
        raise ValueError('historical campaign SHA-256 differs from the approved reference')
    campaign = json.loads(path.read_text(encoding='utf-8'))
    if campaign['status'] != 'complete' or campaign['errors'] or campaign['pending_jobs']:
        raise ValueError('historical reference is not complete')
    marker = json.loads((path.parent / 'COMPLETE').read_text(encoding='utf-8'))
    for key, name in (('campaign_sha256', 'campaign.json'), ('plan_sha256', 'plan.json'),
                      ('preflight_sha256', 'preflight.json')):
        if marker[key] != manifest.sha256_file(path.parent / name):
            raise ValueError('historical closure integrity mismatch')
    for key in ('models', 'image', 'runtime', 'workload', 'network'):
        if config[key] != campaign['protocol'][key]:
            raise ValueError(f'{key} differs from the approved historical reference')
    expected = {k: v for k, v in campaign['input_hashes'].items() if k.endswith('.jsonl')}
    if config['diagnosis']['historical_jsonl_sha256'] != expected or len(expected) != 8:
        raise ValueError('historical JSONL hashes differ from the approved reference')
    return path


def verify_workloads(config, hashes):
    for name, digest in config['diagnosis']['historical_jsonl_sha256'].items():
        if hashes.get(name) != digest:
            raise ValueError(f'{name}: JSONL differs from the historical reference')
