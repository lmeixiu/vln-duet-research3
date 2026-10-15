"""Inspect trained fusion weights; optionally evaluate same-state residual ablation."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def inspect_checkpoint(base, saved, torch):
    original = {(k[7:] if k.startswith('module.') else k): v
                for k, v in base['vln_bert']['state_dict'].items()}
    rows = []
    for name, value in saved['trainable_state_dict'].items():
        value = value.float()
        row = {'name': name, 'finite': bool(torch.isfinite(value).all()),
               'norm': value.norm().item(), 'numel': value.numel()}
        if name in original:
            difference = value - original[name].float()
            row.update(reference='base checkpoint', max_abs_change=difference.abs().max().item(),
                       changed_elements=int((difference != 0).sum()))
        elif '.delta.3.' in name:
            row.update(reference='known zero initialization',
                       max_abs_change=value.abs().max().item(),
                       changed_elements=int((value != 0).sum()))
        else:
            row['reference'] = 'initial random tensor not saved; change unknown'
        rows.append(row)
    optimizer = saved.get('optimizer', {}).get('state', {})
    active = sum(any(torch.is_tensor(v) and bool((v != 0).any())
                     for k, v in state.items() if k in ('exp_avg', 'square_avg', 'momentum_buffer'))
                 for state in optimizer.values())
    return {'mode': saved['mode'], 'epoch': saved.get('epoch'), 'parameters': rows,
            'optimizer_tensors_with_nonzero_moments': active,
            'note': 'Moments are indirect historical evidence, not a current gradient measurement.'}


def compare_outputs(on, off, delta, torch):
    rows = []
    for i in range(on['fused_logits'].shape[0]):
        a, b = on['fused_logits'][i], off['fused_logits'][i]
        valid = torch.isfinite(a) & torch.isfinite(b)
        alpha_on = on['fuse_weights'][i].item()
        alpha_off = off['fuse_weights'][i].item()
        rows.append({'row': i, 'residual': delta[i].item(),
                     'alpha_on': alpha_on, 'alpha_off': alpha_off,
                     'abs_alpha_change': abs(alpha_on-alpha_off),
                     'saturated_on': alpha_on < .01 or alpha_on > .99,
                     'finite_mask_equal': bool(torch.equal(torch.isfinite(a), torch.isfinite(b))),
                     'max_abs_logit_change': (a[valid]-b[valid]).abs().max().item() if valid.any() else None,
                     'action_on': int(a.argmax()), 'action_off': int(b.argmax()),
                     'action_changed': bool(a.argmax() != b.argmax())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='Original complete DUET checkpoint')
    parser.add_argument('--adapter', required=True, help='Trained .pilot.pt file')
    parser.add_argument('--output', required=True, help='NEW directory for diagnostics')
    parser.add_argument('--evaluate', action='store_true', help='Also run GPU/MatterSim evaluation')
    parser.add_argument('--data-root')
    parser.add_argument('--limit-val-examples', type=int, default=64)
    parser.add_argument('--rollout-residual', choices=['on', 'off'],
                        help='Independent complete rollout, instead of same-state diagnosis')
    args = parser.parse_args()
    import torch
    from pilot_checkpoint import read_checkpoint
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('--output must be empty/new to avoid mixing diagnosis runs')
    out.mkdir(parents=True, exist_ok=True)
    saved = read_checkpoint(args.adapter)
    report = inspect_checkpoint(read_checkpoint(args.base), saved, torch)
    (out/'parameter_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in report['parameters']:
        if 'max_abs_change' in row:
            print(row['name'], 'max change:', row['max_abs_change'])
    print('Parameter report:', out/'parameter_report.json')
    if not args.evaluate:
        return
    if not args.data_root:
        parser.error('--evaluate requires --data-root')
    if saved['mode'] == 'off':
        parser.error('Residual ablation requires a language adapter, not mode=off')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'map_nav_src'))
    from models.vilmodel import GlocalTextPathNavCMT
    from r2r.agent import GMapNavAgent
    original_forward = GlocalTextPathNavCMT.forward_navigation_per_step
    original_test = GMapNavAgent.test
    records = []

    def diagnosed_forward(model, *inputs, **kwargs):
        if model.training:
            raise RuntimeError('Counterfactual diagnosis is evaluation-only')
        if args.rollout_residual == 'on':
            with torch.no_grad():
                return original_forward(model, *inputs, **kwargs)
        if args.rollout_residual == 'off':
            hook = model.semantic_fusion_adapter.register_forward_hook(
                lambda m, x, y: torch.zeros_like(y))
            try:
                with torch.no_grad():
                    return original_forward(model, *inputs, **kwargs)
            finally:
                hook.remove()
        captured = []
        adapter = model.semantic_fusion_adapter
        hook = adapter.register_forward_hook(lambda m, x, y: captured.append(y.detach()))
        try:
            with torch.no_grad():
                enabled = original_forward(model, *inputs, **kwargs)
        finally:
            hook.remove()
        hook = adapter.register_forward_hook(lambda m, x, y: torch.zeros_like(y))
        try:
            with torch.no_grad():
                disabled = original_forward(model, *inputs, **kwargs)
        finally:
            hook.remove()
        batch_records = compare_outputs(enabled, disabled, captured[0], torch)
        for row in batch_records:
            row['forward_call'] = diagnosed_forward.calls
        records.extend(batch_records)
        diagnosed_forward.calls += 1
        return enabled  # Actual rollout follows the trained model, never the ablation.

    diagnosed_forward.calls = 0

    def diagnosed_test(agent, *inputs, **kwargs):
        result = original_test(agent, *inputs, **kwargs)
        predictions = agent.get_results()
        (out/'trajectories.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')
        averages, per_instruction = agent.env.eval_metrics(predictions)
        def numeric(value):
            return value.item() if hasattr(value, 'item') else value
        (out/'navigation_metrics.json').write_text(json.dumps(
            {'residual': args.rollout_residual, 'aggregate': averages,
             'per_instruction': per_instruction}, default=numeric, indent=2), encoding='utf-8')
        return result

    config = saved['args']
    forwarded = []
    for key in ('tokenizer', 'features', 'image_feat_size', 'angle_feat_size',
                'obj_feat_size', 'num_l_layers', 'num_pano_layers', 'num_x_layers',
                'max_instr_len', 'max_action_len', 'batch_size', 'seed'):
        if key in config:
            forwarded.extend(['--'+key, str(config[key])])
    for key in ('enc_full_graph', 'graph_sprels', 'act_visited_nodes', 'no_backtrack'):
        if config.get(key):
            forwarded.append('--'+key)
    sys.argv = ['run_semantic_pilot.py', '--mode', saved['mode'],
                '--resume_file', args.base, '--pilot-checkpoint', args.adapter,
                '--root_dir', args.data_root, '--output_dir', str(out/'evaluation'),
                '--fusion', 'dynamic', '--iters', '0', '--eval_first',
                '--limit-val-examples', str(args.limit_val_examples)] + forwarded
    GlocalTextPathNavCMT.forward_navigation_per_step = diagnosed_forward
    GMapNavAgent.test = diagnosed_test
    try:
        runpy.run_path(str(Path(__file__).with_name('run_semantic_pilot.py')), run_name='__main__')
    finally:
        GlocalTextPathNavCMT.forward_navigation_per_step = original_forward
        GMapNavAgent.test = original_test
    if args.rollout_residual is not None:
        print('Independent rollout complete: residual=%s; results: %s' % (args.rollout_residual, out))
        return
    (out/'same_state_steps.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    n = len(records)
    summary = {'forward_rows': n,
               'action_change_rate': sum(r['action_changed'] for r in records)/n if n else None,
               'mean_abs_alpha_change': sum(r['abs_alpha_change'] for r in records)/n if n else None,
               'mean_abs_residual': sum(abs(r['residual']) for r in records)/n if n else None,
               'saturated_rate': sum(r['saturated_on'] for r in records)/n if n else None,
               'note': 'Rows include ended/padded and repeated evaluation batch slots; rates are diagnostics, not episode metrics. Off retains the TRAINED original gate. No retraining; no ground-truth gradient measured.'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
