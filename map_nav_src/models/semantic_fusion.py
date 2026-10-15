"""Language-conditioned residual for the original DUET scalar gate."""
import math
import torch
from torch import nn


class SemanticFusionAdapter(nn.Module):
    """Zero-initialized language residual for DUET's scalar fusion gate."""
    MODES = {"mean", "token_attention", "semantic_chunks", "equal_chunks"}

    def __init__(self, hidden_size, mode, bottleneck=128):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("Unsupported semantic fusion mode: %s" % mode)
        self.mode = mode
        self.query = nn.Linear(hidden_size * 2, bottleneck, bias=False)
        self.key = nn.Linear(hidden_size, bottleneck, bias=False)
        self.value = nn.Linear(hidden_size, bottleneck, bias=False)
        self.delta = nn.Sequential(
            nn.Linear(hidden_size * 2 + bottleneck, bottleneck),
            nn.ReLU(),
            nn.LayerNorm(bottleneck, eps=1e-12),
            nn.Linear(bottleneck, 1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def _chunk_features(self, txt_embeds, txt_chunk_ids):
        if txt_chunk_ids is None:
            raise ValueError(
                "This mode needs txt_chunk_ids; run through tools/run_semantic_pilot.py."
            )
        valid = txt_chunk_ids >= 0
        if not valid.any(dim=1).all():
            raise ValueError("Each instruction must contain at least one chunk token")
        chunk_count = int(txt_chunk_ids.max().item()) + 1
        slots = torch.arange(chunk_count, device=txt_embeds.device)
        membership = txt_chunk_ids.unsqueeze(1) == slots.view(1, -1, 1)
        membership_f = membership.to(dtype=txt_embeds.dtype)
        counts = membership_f.sum(dim=-1).clamp_min(1.0)
        features = torch.bmm(membership_f, txt_embeds) / counts.unsqueeze(-1)
        return features, membership.any(dim=-1)

    def _attend(self, branch_state, features, feature_mask):
        query = self.query(branch_state).unsqueeze(1)
        keys = self.key(features)
        scores = (query * keys).sum(dim=-1) / math.sqrt(keys.size(-1))
        scores = scores.masked_fill(~feature_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), self.value(features)).squeeze(1)

    def forward(self, branch_state, txt_embeds, txt_masks, txt_chunk_ids=None):
        if txt_chunk_ids is None:
            raise ValueError("Language fusion requires aligned content masks from pilot launcher")
        txt_masks = txt_masks & (txt_chunk_ids >= 0)
        if not txt_masks.any(dim=1).all():
            raise ValueError("Instruction has no usable language tokens")
        if self.mode == "mean":
            weights = txt_masks.to(dtype=txt_embeds.dtype).unsqueeze(-1)
            language_state = self.value((txt_embeds * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0))
        elif self.mode == "token_attention":
            language_state = self._attend(branch_state, txt_embeds, txt_masks)
        else:
            group_ids = txt_chunk_ids.masked_fill(~txt_masks, -1)
            features, feature_mask = self._chunk_features(txt_embeds, group_ids)
            language_state = self._attend(branch_state, features, feature_mask)
        return self.delta(torch.cat([branch_state, language_state], dim=-1))

