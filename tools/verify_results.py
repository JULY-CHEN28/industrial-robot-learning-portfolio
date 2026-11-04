"""Recompute published counts; optionally verify the paired raw state arrays."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('position_reached', 'position_hold', 'pose_reached', 'pose_hold')


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(raw=False):
    comparison = read('results/isaaclab/comparison.json')
    for model in ('author', 'ours44'):
        aggregate = dict.fromkeys(METRICS, 0)
        for seed in comparison['test_seeds']:
            base = f'results/isaaclab/paired/{model}_{seed}'
            report = read(f'{base}/summary.json')
            windows = read(f'{base}/windows.json')
            counts = {k: sum(int(w[k]) for w in windows) for k in METRICS}
            require(len(windows) == report['complete_target_windows'] == 320, base)
            require(counts == report['counts'], f'Window counts differ: {base}')
            for k in METRICS:
                aggregate[k] += counts[k]
            if raw:
                verify_raw(base, report, windows)
        expected = comparison['aggregate'][model]
        require(aggregate == expected['counts'], f'Aggregate differs: {model}')
        for k in METRICS:
            require(abs(aggregate[k] / 640 * 100 - expected['rates_percent'][k]) < 1e-9, k)
    if raw:
        import numpy as np
        for seed in comparison['test_seeds']:
            folder = ROOT / 'results/isaaclab/paired'
            with np.load(folder / f'author_{seed}/states.npz') as a, np.load(folder / f'ours44_{seed}/states.npz') as b:
                for key in ('commands', 'reset'):
                    require(np.array_equal(a[key], b[key]), f'Paired {key} differs')
    kinova = [read(f'results/kinova_mujoco/{name}.json') for name in ('clearance_short_410000', 'clearance_long_510000')]
    require(sum(d['target_windows'] for d in kinova) == 600, 'Kinova targets')
    require(sum(d['reached_windows'] for d in kinova) == 592, 'Kinova reach')
    require(sum(d['held_windows'] for d in kinova) == 579, 'Kinova hold')
    jaka = [read(f'results/jaka_mujoco/{name}.json') for name in ('continuous_60s_510000', 'continuous_300s_610000')]
    for d in jaka:
        require(len(d['per_target']) == d['targets'], 'JAKA target count')
        for key in ('reached', 'held_0_5s'):
            require(sum(int(t[key]) for t in d['per_target']) == d[key] == d['targets'], key)
    require(sum(d['targets'] for d in jaka) == 675, 'JAKA total')
    mean_mm = sum(d['mean_final_error_m'] * d['targets'] for d in jaka) / 675 * 1000
    require(round(mean_mm, 2) == 2.98, 'JAKA final mean')
    cases = read('results/jaka_mujoco/robustness_confirm_810000/summary.json')
    require({d['case']: d['held_0_5s'] for d in cases} == {
        'nominal': 60, 'command_delay_96ms': 60, 'target_bias_30mm': 16,
        'combined_mild': 60, 'combined_stress': 13}, 'Disturbance summary')
    for name, digest in read('SHA256SUMS.json').items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Hash differs: {name}')
    print('PASS: paired counts, aggregates, continuous targets, perturbations, publication hashes')
    if raw:
        print('PASS: raw-state window reconstruction, continuous holds, paired targets and resets')


def verify_raw(base, report, windows):
    import numpy as np
    with np.load(ROOT / base / 'states.npz') as data:
        p, r, goals, reset = (data[k] for k in ('position_m', 'orientation_rad', 'commands', 'reset'))
        require(p.shape == r.shape == (720, 64), 'Unexpected raw shape')
        require(np.isfinite(p).all() and np.isfinite(r).all(), 'Nonfinite error')
        counts = dict.fromkeys(METRICS, 0)
        reconstructed = []
        for env in range(64):
            changes = np.any(goals[1:, env] != goals[:-1, env], axis=1) | reset[1:, env]
            bounds = np.r_[0, np.flatnonzero(changes) + 1, len(p)]
            for a, b in zip(bounds[:-1], bounds[1:]):
                if (b-a) * report['dt'] < 4 - 1.5 * report['dt']:
                    continue
                reconstructed.append((env, int(a), int(b-a)))
                position = p[a:b, env] < report['position_threshold_m']
                pose = position & (r[a:b, env] < report['orientation_threshold_rad'])
                for mask, reach, hold in ((position, 'position_reached', 'position_hold'), (pose, 'pose_reached', 'pose_hold')):
                    counts[reach] += int(mask.any())
                    counts[hold] += int(np.any(np.convolve(mask.astype(int), np.ones(16, dtype=int), 'valid') == 16))
        require(counts == report['counts'], f'Raw counts differ: {base}')
        require(reconstructed == [(w['env'], w['start_step'], w['samples']) for w in windows], 'Window boundaries differ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', action='store_true', help='also check raw NPZ arrays; requires NumPy')
    verify(parser.parse_args().raw)
