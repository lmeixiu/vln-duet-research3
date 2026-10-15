"""CPU checks: python tools/test_semantic_pilot.py (torch tests skip if absent)."""
import importlib.util
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'map_nav_src'))
from run_semantic_pilot import InstructionChunker


class OffsetTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        spans = [m.span() for m in re.finditer(r'\w+|[^\w\s]', text)]
        return {'input_ids': [101] + list(range(1000, 1000 + len(spans))) + [102],
                'offset_mapping': [(0, 0)] + spans + [(0, 0)]}


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = OffsetTokenizer()
        self.chunker = InstructionChunker(self.tokenizer)
        self.text = 'Walk past the sofa, turn left at the table, and stop near the kitchen.'
        self.ids = self.tokenizer(self.text)['input_ids']

    def test_truncated_annotations_without_sep(self):
        groups = self.chunker.token_ids(self.text, self.ids[:10], 'semantic_chunks')
        self.assertEqual(len(groups), 10)
        self.assertGreaterEqual(groups[-1], 0)

    def test_matched_count_and_content(self):
        semantic = self.chunker.token_ids(self.text, self.ids, 'semantic_chunks')
        equal = self.chunker.token_ids(self.text, self.ids, 'equal_chunks')
        self.assertEqual(max(semantic), 2)
        self.assertEqual(max(semantic), max(equal))
        self.assertEqual([x >= 0 for x in semantic], [x >= 0 for x in equal])
        self.assertEqual(semantic[0], -1)
        self.assertEqual(semantic[-1], -1)

    def test_reject_mismatched_annotation(self):
        with self.assertRaises(ValueError):
            self.chunker.token_ids(self.text, [999] + self.ids[1:], 'semantic_chunks')


HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'PyTorch required for numerical checks')
class NumericalTests(unittest.TestCase):
    def test_diagnostic_masks_and_action_changes(self):
        import torch
        from diagnose_semantic_pilot import compare_outputs
        on = {'fused_logits': torch.tensor([[1., 2., -float('inf')], [3., 0., -float('inf')]]),
              'fuse_weights': torch.tensor([[.5], [.999]])}
        off = {'fused_logits': torch.tensor([[2., 1., -float('inf')], [4., 0., -float('inf')]]),
               'fuse_weights': torch.tensor([[.4], [.8]])}
        rows = compare_outputs(on, off, torch.tensor([[.2], [.3]]), torch)
        self.assertTrue(rows[0]['action_changed'])
        self.assertFalse(rows[1]['action_changed'])
        self.assertTrue(rows[1]['saturated_on'])
        self.assertEqual(rows[0]['max_abs_logit_change'], 1.)
        self.assertTrue(rows[0]['finite_mask_equal'])

    def test_zero_residual_gradient_and_mask(self):
        import torch
        from models.semantic_fusion import SemanticFusionAdapter
        for mode in SemanticFusionAdapter.MODES:
            torch.manual_seed(4)
            model = SemanticFusionAdapter(8, mode, bottleneck=4)
            state, text = torch.randn(2, 16), torch.randn(2, 6, 8)
            ids = torch.tensor([[-1, 0, 0, 1, 1, -1], [-1, 0, 0, 0, -1, -1]])
            mask = ids >= 0
            self.assertTrue(torch.equal(model(state, text, mask, ids), torch.zeros(2, 1)))
            optimizer = torch.optim.SGD(model.parameters(), lr=.05)
            for _ in range(3):
                optimizer.zero_grad()
                (model(state, text, mask, ids) - 1).square().mean().backward()
                optimizer.step()
            self.assertGreater(model.value.weight.grad.abs().sum().item(), 0)
            output = model(state, text, mask, ids)
            corrupted = text.clone()
            corrupted[~mask] = 10000
            torch.testing.assert_close(output, model(state, corrupted, mask, ids))

    def test_base_and_adapter_round_trip(self):
        import torch
        from torch import nn
        from models.semantic_fusion import SemanticFusionAdapter
        from pilot_checkpoint import load_base_and_adapter
        model = nn.Module()
        model.vln_bert = nn.Module()
        model.vln_bert.backbone = nn.Linear(8, 8)
        model.vln_bert.sap_fuse_linear = nn.Linear(16, 1)
        model.vln_bert.semantic_fusion_adapter = SemanticFusionAdapter(8, 'semantic_chunks', 4)
        for p in model.vln_bert.backbone.parameters():
            p.requires_grad_(False)
        with tempfile.TemporaryDirectory() as temp:
            base, saved = Path(temp) / 'base.pt', Path(temp) / 'adapter.pt'
            torch.save({'vln_bert': {'state_dict': {k: v for k, v in model.state_dict().items()
                        if 'semantic_fusion_adapter' not in k}, 'epoch': 20000}}, base)
            load_base_and_adapter(model, base, None, 'semantic_chunks')
            state, text = torch.randn(1, 16), torch.randn(1, 4, 8)
            ids = torch.tensor([[-1, 0, 1, -1]])
            adapter = model.vln_bert.semantic_fusion_adapter
            with torch.no_grad():
                adapter.delta[-1].weight.fill_(.1)
            expected = adapter(state, text, ids >= 0, ids).detach()
            torch.save({'format_version': 2, 'mode': 'semantic_chunks',
                        'trainable_state_dict': {k: p.detach().clone() for k, p in model.named_parameters()
                                                if p.requires_grad}}, saved)
            load_base_and_adapter(model, base, saved, 'semantic_chunks')
            torch.testing.assert_close(expected, adapter(state, text, ids >= 0, ids))
            with self.assertRaises(ValueError):
                load_base_and_adapter(model, base, saved, 'token_attention')


if __name__ == '__main__':
    if '--require-torch' in sys.argv:
        sys.argv.remove('--require-torch')
        if not HAS_TORCH:
            raise SystemExit('PyTorch is required; numerical verification was NOT performed')
    unittest.main()
