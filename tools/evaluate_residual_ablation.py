"""Run separate ON/OFF navigation episodes from the same trained checkpoint."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def compare(on, off, paths_on, paths_off):
    def index(metrics):
        values = metrics['per_instruction']
        ids = values['instr_id']
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate instruction IDs in metrics')
        return {iid: {k: v[i] for k, v in values.items()} for i, iid in enumerate(ids)}
    a, b = index(on), index(off)
    pa = {r['instr_id']: r['trajectory'] for r in paths_on}
    pb = {r['instr_id']: r['trajectory'] for r in paths_off}
    if not a or set(a) != set(b) or set(a) != set(pa) or set(a) != set(pb):
        raise ValueError('ON/OFF instruction sets must match and be nonempty')
    rows = []
    for iid in sorted(a):
        sa, sb = bool(a[iid]['success']), bool(b[iid]['success'])
        rows.append({'instr_id': iid, 'success_on': sa, 'success_off': sb,
                     'helped': sa and not sb, 'hurt': sb and not sa,
                     'trajectory_changed': pa[iid] != pb[iid],
                     'spl_on': a[iid]['spl'], 'spl_off': b[iid]['spl'],
                     'nav_error_on': a[iid]['nav_error'], 'nav_error_off': b[iid]['nav_error']})
    summary = {'instructions': len(rows), 'on': on['aggregate'], 'off': off['aggregate'],
               'on_minus_off': {k: on['aggregate'][k]-off['aggregate'][k] for k in on['aggregate']},
               'helped_count': sum(r['helped'] for r in rows),
               'hurt_count': sum(r['hurt'] for r in rows),
               'changed_trajectory_count': sum(r['trajectory_changed'] for r in rows),
               'note': 'Both runs retain the same TRAINED original gate. This is a post-training residual ablation, not the independently trained off baseline. SR/SPL aggregate deltas are percentage points; per-instruction SPL is 0..1.'}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit-val-examples', type=int, default=64)
    args = parser.parse_args()
    if args.limit_val_examples < 0:
        parser.error('--limit-val-examples must be nonnegative (0 = full split)')
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('--output must be new/empty')
    out.mkdir(parents=True, exist_ok=True)
    tool = Path(__file__).with_name('diagnose_semantic_pilot.py').resolve()
    for mode in ('on', 'off'):
        # Separate processes release GPU memory and reset model/environment state.
        subprocess.run([sys.executable, str(tool), '--base', str(Path(args.base).resolve()),
                        '--adapter', str(Path(args.adapter).resolve()), '--data-root', str(Path(args.data_root).resolve()),
                        '--output', str(out/mode), '--evaluate', '--rollout-residual', mode,
                        '--limit-val-examples', str(args.limit_val_examples)],
                       cwd=str(tool.parents[1]/'map_nav_src'), check=True)
    def read(mode, filename):
        return json.loads((out/mode/filename).read_text(encoding='utf-8'))
    if read('on', 'evaluation/pilot_subsets.json') != read('off', 'evaluation/pilot_subsets.json'):
        raise ValueError('ON/OFF subset manifests differ')
    summary, rows = compare(read('on', 'navigation_metrics.json'), read('off', 'navigation_metrics.json'),
                            read('on', 'trajectories.json'), read('off', 'trajectories.json'))
    (out/'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out/'per_instruction_comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Comparison saved to:', out/'comparison.json')


if __name__ == '__main__':
    main()
