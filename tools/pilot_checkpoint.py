"""Strict base-plus-adapter loading shared by the launcher and CPU tests."""
import inspect
import torch


def read_checkpoint(path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def load_base_and_adapter(model, base_path, adapter_path, mode):
    base = read_checkpoint(base_path)
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in base['vln_bert']['state_dict'].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any('semantic_fusion_adapter.' not in k for k in missing):
        raise ValueError('Incompatible base checkpoint: missing=%s unexpected=%s' % (missing, unexpected))
    adapter = model.vln_bert.semantic_fusion_adapter
    if adapter is not None:
        # Reset AFTER all HF/model/base loading has completed.
        torch.nn.init.zeros_(adapter.delta[-1].weight)
        torch.nn.init.zeros_(adapter.delta[-1].bias)
    if adapter_path:
        saved = read_checkpoint(adapter_path)
        if saved.get('format_version') != 2 or saved['mode'] != mode:
            raise ValueError('Adapter version/mode does not match this run')
        expected = {k for k, p in model.named_parameters() if p.requires_grad}
        if set(saved['trainable_state_dict']) != expected:
            raise ValueError('Adapter parameter set does not match this model')
        model.load_state_dict(saved['trainable_state_dict'], strict=False)
    print('Loaded base checkpoint; pilot update counter starts at zero')
