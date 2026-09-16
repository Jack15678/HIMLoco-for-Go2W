"""Produce per-change coverage signals and per-task failures, without a composite score."""
import argparse
import json
from pathlib import Path
import numpy as np


def report(run, evaluations, window=200):
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('No completed training iterations')
    last = rows[-1]
    first = rows[max(0, len(rows)-window-1)] if len(rows) > window else None
    if first is None and (run/'resume_verification.json').exists():
        initial = json.loads((run/'resume_verification.json').read_text())
        first = dict(iteration=initial['iteration'], curriculum=initial['initial_curriculum'])
    current = last['curriculum']
    def difference(key):
        values = np.asarray(current[key], dtype=float)
        return values if first is None else values-np.asarray(first['curriculum'][key])
    seconds, target, squared, good, samples = [difference(k) for k in
        ['total_seconds', 'total_target_seconds', 'total_squared_error', 'total_good', 'samples']]
    cells = []
    for i, terrain in enumerate(current['terrain_names']):
        for j, mode in enumerate(current['mode_names']):
            n = seconds[i, j]
            cells.append(dict(terrain=terrain, mode=mode, seconds=n,
                target_contact_seconds=target[i, j],
                sample_counts_normal_extended=samples[i, j].tolist(),
                rmse=np.sqrt(squared[i, j]/n).tolist() if n > 0 else None,
                simultaneous_success_fraction=float(good[i, j]/n) if n > 0 else None,
                coverage='not_sampled' if n == 0 else ('insufficient_target_contact' if target[i,j] < 2 else 'exercised')))
    cases = []
    if evaluations and evaluations.exists():
        for path in sorted(evaluations.rglob('summary.json')):
            data = json.loads(path.read_text())
            if data.get('protocol') != 's10-terrain-task-v2' or 'segments' not in data:
                continue
            probe, hold = data['segments'][1:3]
            if data['end_reason'] != 'completed':
                status = 'failed_or_interrupted'
            elif probe.get('evidence') == 'insufficient_target_contact':
                status = 'insufficient_target_contact'
            elif not probe.get('passed', False):
                status = 'tracking_deficit'
            elif not hold.get('passed', False):
                status = 'braking_or_holding_deficit'
            else:
                status = 'passed_fixed_case'
            cases.append(dict(path=str(path.relative_to(evaluations)), scene=data['scene'], probe=data['probe'],
                seed=data['seed'], status=status, task=probe, stopping=hold,
                tilt=data['max_tilt_degrees'], saturation=data['pd_saturation_fraction']))
    return dict(iteration=last['iteration'], window_start_iteration=first['iteration'] if first else 0,
        speed_limits=current['speed_limits'], terrain_mode_cells=cells,
        old_new_grade_matrix=difference('old_new_disagreements').tolist(),
        promotions=difference('up_count').tolist(), demotions=difference('down_count').tolist(),
        task_evaluations=cases,
        evaluation_status='available' if cases else 'not_run_or_no_v2_results',
        interpretation='Coverage and same-trajectory grade changes diagnose implementation effects. Fixed-case behavior diagnoses deficits. A single combined run does not identify isolated causal contributions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--evaluations', type=Path)
    parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.window < 1:
        parser.error('window must be positive')
    result = report(args.run, args.evaluations, args.window)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(iteration=result['iteration'], evaluation_status=result['evaluation_status'],
        deficits=[{k: x[k] for k in ['scene', 'probe', 'status']} for x in result['task_evaluations'] if x['status'] != 'passed_fixed_case']), indent=2))
