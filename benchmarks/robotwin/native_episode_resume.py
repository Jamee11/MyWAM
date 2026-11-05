"""Opt-in native episode recovery; default evaluation is unchanged."""
from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
from pathlib import Path
import random
import re
import textwrap

ANSI = rb'(?:\x1b\[[0-9;]*m)*'
ROW = re.compile(rb'Success rate:\s*' + ANSI + rb'(\d+)/(\d+).*?current seed:\s*' + ANSI + rb'(\d+)[^\r\n]*(?:\r?\n|$)')
ERRORS = (b'OIDN Error', b'ErrorDeviceLost', b'VK_ERROR_DEVICE_LOST',
          b'illegal memory access', b'cannot create buffer', b'failed to create buffer')


def read_prefix(directory: Path, start_seed: int, total: int = 70) -> dict:
    """Only complete cumulative records strictly before the first renderer error."""
    directory = Path(directory)
    raw = (directory / 'client.log').read_bytes()
    positions = [raw.find(marker) for marker in ERRORS if marker in raw]
    limit = min(positions, default=len(raw))
    rows, end = [], 0
    for match in ROW.finditer(raw, 0, limit):
        good, count, seed = map(int, match.groups())
        previous = rows[-1]['successes'] if rows else 0
        if (count != len(rows) + 1 or count > total or good-previous not in (0, 1)
                or seed < start_seed or (rows and seed <= rows[-1]['seed'])):
            break
        rows.append(dict(successes=good, count=count, seed=seed))
        end = match.end()
    rng_state = None
    journal = directory / 'episode_journal.jsonl'
    if rows and journal.exists():
        for line in journal.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(record.get(k) == rows[-1][k] for k in ('successes', 'count', 'seed')):
                rng_state = record.get('python_rng_state')
    carried = directory / 'resume_prefix.json'
    if rows and rng_state is None and carried.exists():
        saved = json.loads(carried.read_text())
        if saved['rows'] == rows:
            rng_state = saved.get('python_rng_state')
    return dict(source=str(directory.resolve()), source_prefix_bytes=end,
                source_prefix_sha256=hashlib.sha256(raw[:end]).hexdigest(),
                rows=rows, python_rng_state=rng_state,
                legacy_rng_boundary=bool(rows and rng_state is None))


def choose_prefix(run: Path, task: str, mode: str, current: Path, start_seed: int, total: int = 70) -> dict:
    """Choose furthest verified progress, never choose by its success rate."""
    run, current = Path(run), Path(current).resolve()
    state = json.loads((run / 'state.json').read_text())
    condition = next(c for c in state['conditions'] if c['task'] == task and c['mode'] == mode)
    candidates = []
    for attempt in condition['attempts']:
        directory = Path(attempt['output']).resolve()
        if directory == current or not (directory / 'client.log').exists():
            continue
        candidate = read_prefix(directory, start_seed, total)
        if candidate['rows']:
            if (directory / 'resume_prefix.json').exists():
                verify_provenance(directory)
            candidates.append(candidate)
    best = max(candidates, key=lambda c: len(c['rows']), default=dict(rows=[], python_rng_state=None, legacy_rng_boundary=False))
    return dict(best, task=task, mode=mode, start_seed=start_seed, total=total,
                checkpoint_sha256=state.get('checkpoint_sha256'))


def verify_provenance(directory: Path, seen=None) -> dict | None:
    """Verify carried records against retained original log bytes, recursively."""
    directory = Path(directory).resolve()
    path = directory / 'resume_prefix.json'
    if not path.exists():
        return None
    seen = set() if seen is None else seen
    if directory in seen:
        raise ValueError('Cyclic episode recovery provenance')
    seen.add(directory)
    prefix = json.loads(path.read_text())
    if not prefix['rows']:
        return prefix
    source = Path(prefix['source'])
    raw = (source / 'client.log').read_bytes()[:prefix['source_prefix_bytes']]
    if hashlib.sha256(raw).hexdigest() != prefix['source_prefix_sha256']:
        raise ValueError('Recovered source log prefix changed')
    checked = read_prefix(source, prefix['start_seed'], prefix['total'])
    if checked['rows'][:len(prefix['rows'])] != prefix['rows']:
        raise ValueError('Recovered rows do not match clean source records')
    own = read_prefix(directory, prefix['start_seed'], prefix['total'])
    if own['rows'][:len(prefix['rows'])] != prefix['rows']:
        raise ValueError('Carried rows do not match this attempt log')
    verify_provenance(source, seen)
    return prefix


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def install(module, attempt: Path, prefix: dict) -> None:
    """Keep the native episode count and language pool; restore prior progress."""
    attempt = Path(attempt)
    original = module.eval_policy
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or original.__closure__:
        raise ValueError('Expected the unwrapped native eval_policy')
    points = [i for i, node in enumerate(function.body)
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'now_seed' for t in node.targets)
              and isinstance(node.value, ast.Name) and node.value.id == 'st_seed']
    if len(points) != 1:
        raise ValueError('Unknown native evaluator initialization')
    initialization = ast.parse('now_id, succ_seed, suc_test_seed_list, now_seed = _openwam_resume_begin(TASK_ENV, args, st_seed, test_num)').body[0]
    function.body.insert(points[0] + 1, initialization)
    context = {}
    base_print = module.__dict__.get('print', builtins.print)

    def begin(env, args, start_seed, test_num):
        if test_num != prefix['total'] or start_seed != prefix['start_seed']:
            raise ValueError('Recovery requires the original native request and seed boundary')
        context['env'] = env
        rows = prefix['rows']
        count = len(rows)
        env.suc = rows[-1]['successes'] if rows else 0
        env.test_num = count
        (attempt / 'resume_prefix.json').write_text(json.dumps(prefix, indent=2) + '\n')
        base_print(f'EPISODE_RESUME preserved={count} remaining={test_num-count} language_pool={test_num} legacy_rng_boundary={prefix["legacy_rng_boundary"]}', flush=True)
        # These are explicitly carried records, with immutable byte provenance above.
        for row in rows:
            base_print(f'Success rate: {row["successes"]}/{row["count"]} => carried completed episode, current seed: {row["seed"]}', flush=True)
        if prefix.get('python_rng_state') is not None:
            random.setstate(_tuples(prefix['python_rng_state']))
        return count, count, [r['seed'] for r in rows], rows[-1]['seed'] + 1 if rows else start_seed

    def journal_print(*args, **kwargs):
        text = ' '.join(str(arg) for arg in args).encode()
        match = ROW.search(text + b'\n')
        if match and context:
            good, count, seed = map(int, match.groups())
            env = context['env']
            record = dict(successes=good, count=count, seed=seed,
                          instruction=env.get_instruction(), python_rng_state=random.getstate())
            with (attempt / 'episode_journal.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        base_print(*args, **kwargs)

    module.__dict__['_openwam_resume_begin'] = begin
    module.__dict__['print'] = journal_print
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), module.__dict__, namespace)
    module.eval_policy = namespace[function.name]
