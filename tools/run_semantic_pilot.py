#!/usr/bin/env python3
"""Run a low-cost R2R fusion pilot on a fine-tuned DUET checkpoint.

All standard DUET arguments (including --root_dir, --resume_file, --batch_size,
--iters, and --output_dir) are forwarded to the official R2R entry point.
"""
import argparse
from pathlib import Path
import re
import sys
import json
import time


class InstructionChunker:
    """Map simple action/subgoal clauses onto the existing tokenizer pieces."""
    _boundary = re.compile(
        r"[,;!?]+|\.\s*|\b(?:and\s+then|then|after\s+that|next)\b\s*|"
        r"\band\s+(?=(?:turn|walk|go|head|continue|proceed|move|pass|cross|"
        r"enter|exit|leave|stop|take|keep|follow|climb|reach)\b)",
        re.IGNORECASE,
    )

    def __init__(self, tokenizer, max_chunks=0):
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("Chunk alignment requires a fast Hugging Face tokenizer.")
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks

    @classmethod
    def spans(cls, text):
        spans, start = [], 0
        for boundary in cls._boundary.finditer(text):
            end = boundary.start()
            if text[start:end].strip():
                spans.append((start, end))
            start = boundary.end()
        if text[start:].strip():
            spans.append((start, len(text)))
        return spans or [(0, len(text))]

    def token_ids(self, instruction, expected_ids, mode):
        expected_ids = [int(token_id) for token_id in expected_ids]
        encoded = self.tokenizer(
            instruction,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        actual_ids = [int(token_id) for token_id in encoded["input_ids"][:len(expected_ids)]]
        if actual_ids != expected_ids:
            mismatch = next(
                (i for i, pair in enumerate(zip(actual_ids, expected_ids)) if pair[0] != pair[1]),
                min(len(actual_ids), len(expected_ids)),
            )
            raise ValueError(
                "Tokenizer output differs from the pre-encoded R2R instruction "
                "for %r at token %d (tokenizer=%d tokens, data=%d). Check that "
                "--tokenizer matches the R2R annotation encoding."
                % (instruction[:80], mismatch, len(actual_ids), len(expected_ids))
            )
        offsets = encoded["offset_mapping"][:len(expected_ids)]
        content_positions = [i for i, (a, b) in enumerate(offsets) if b > a]
        ids = [-1] * len(expected_ids)
        if not content_positions:
            return ids

        spans = self.spans(instruction)
        chunk_tokens = [[] for _ in spans]
        for position in content_positions:
            token_start, token_end = offsets[position]
            midpoint = (token_start + token_end) / 2.0
            chosen = next(
                (chunk_index for chunk_index, (start, end) in enumerate(spans)
                 if start <= midpoint < end),
                None,
            )
            if chosen is not None:
                chunk_tokens[chosen].append(position)

        # Remove empty clauses and compact the ids while keeping route order.
        chunk_index = 0
        for positions in chunk_tokens:
            if not positions:
                continue
            for position in positions:
                ids[position] = chunk_index
            chunk_index += 1
        if chunk_index == 0:
            for position in content_positions:
                ids[position] = 0
        # Controls share exactly the same content tokens, excluding boundaries.
        if mode == "equal_chunks":
            positions = [i for i, group in enumerate(ids) if group >= 0]
            count = max(ids) + 1
            for rank, position in enumerate(positions):
                ids[position] = min(count - 1, rank * count // len(positions))
        return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--mode",
        choices=["off", "mean", "token_attention", "semantic_chunks", "equal_chunks"],
        required=True,
        help="off=DUET gate; mean=whole instruction; token_attention=token control; "
             "semantic_chunks=proposed method; equal_chunks=granularity control",
    )
    parser.add_argument("--limit-train-examples", type=int, default=64)
    parser.add_argument("--limit-val-examples", type=int, default=64)
    parser.add_argument("--pilot-checkpoint", help="Load saved adapter after the base DUET checkpoint")
    pilot, duet_argv = parser.parse_known_args()
    if pilot.limit_train_examples < 0 or pilot.limit_val_examples < 0:
        parser.error("example limits must be non-negative")

    sys.argv = [sys.argv[0]] + duet_argv
    for flag, value in (("--iters", "100"), ("--log_every", "100")):
        if not any(arg == flag or arg.startswith(flag + "=") for arg in duet_argv):
            sys.argv.extend([flag, value])
    map_nav_src = Path(__file__).resolve().parents[1] / "map_nav_src"
    sys.path.insert(0, str(map_nav_src))
    from r2r.parser import parse_args
    from models.vlnbert_init import get_tokenizer
    from r2r.agent import GMapNavAgent
    import torch

    args = parse_args()
    args.semantic_fusion_mode = pilot.mode
    args.pilot_train_examples = pilot.limit_train_examples
    args.pilot_val_examples = pilot.limit_val_examples
    args.pilot_unseen_only = True
    args.pilot_mode = pilot.mode
    if not args.resume_file or not Path(args.resume_file).is_file():
        parser.error("--resume_file must point to an existing fine-tuned DUET checkpoint")
    if args.dataset != 'r2r' or args.train_alg != 'imitation' or args.aug is not None:
        parser.error("This pilot supports R2R imitation learning without augmented data")
    if args.resume_optimizer:
        parser.error("Use fresh pilot optimizers; do not pass --resume_optimizer")
    if args.iters < 0 or args.log_every <= 0:
        parser.error("--iters must be nonnegative and --log_every positive")
    if args.iters == 0:
        args.eval_first = True
    if args.world_size != 1:
        parser.error("The low-cost pilot launcher supports one GPU (set --world_size 1).")

    if args.semantic_fusion_mode != "off":
        if args.fusion != "dynamic":
            parser.error("semantic chunk experiments require --fusion dynamic")
        chunker = InstructionChunker(get_tokenizer(args))
    else:
        chunker = None

    class FrozenFusionAgent(GMapNavAgent):
        def _build_model(self):
            super()._build_model()
            self._chunk_id_cache = {}
            nav_model = self.vln_bert.vln_bert
            for parameter in self.vln_bert.parameters():
                parameter.requires_grad_(False)
            if nav_model.sap_fuse_linear is None:
                raise ValueError("Pass --fusion dynamic to train the DUET fusion gate")
            for parameter in nav_model.sap_fuse_linear.parameters():
                parameter.requires_grad_(True)
            if nav_model.semantic_fusion_adapter is not None:
                for parameter in nav_model.semantic_fusion_adapter.parameters():
                    parameter.requires_grad_(True)
            self.vln_bert.freeze_navigation_backbone = True

            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            count = sum(p.numel() for p in self.vln_bert.parameters() if p.requires_grad)
            print("Trainable fusion parameters: %d" % count)

        def load(self, path):
            from pilot_checkpoint import load_base_and_adapter
            load_base_and_adapter(self.vln_bert, path, pilot.pilot_checkpoint, pilot.mode)
            # A pilot is a new adaptation budget, independent of base epoch.
            return 0

        def _language_variable(self, obs):
            inputs = super()._language_variable(obs)
            if chunker is not None:
                max_length = inputs["txt_ids"].size(1)
                batch_ids = torch.full(
                    (len(obs), max_length), -1, dtype=torch.long,
                    device=inputs["txt_ids"].device,
                )
                for row, observation in enumerate(obs):
                    cache_key = observation["instr_id"]
                    group_ids = self._chunk_id_cache.get(cache_key)
                    if group_ids is None:
                        instruction_ids = observation["instr_encoding"][:max_length]
                        group_ids = chunker.token_ids(
                            observation["instruction"], instruction_ids, pilot.mode
                        )
                        self._chunk_id_cache[cache_key] = group_ids
                    batch_ids[row, :len(group_ids)] = torch.as_tensor(
                        group_ids, dtype=torch.long, device=batch_ids.device
                    )
                inputs["txt_chunk_ids"] = batch_ids
            return inputs

        def save(self, epoch, path):
            # Save only the trained gate/adapters. The full fine-tuned DUET
            # checkpoint remains the reference and is supplied at launch.
            trainable = {
                name: parameter.detach().cpu()
                for name, parameter in self.vln_bert.named_parameters()
                if parameter.requires_grad
            }
            output_path = path + ".pilot.pt"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format_version": 2,
                "epoch": int(epoch) + 1,
                "mode": pilot.mode,
                "base_checkpoint": args.resume_file,
                "trainable_state_dict": trainable,
                "optimizer": self.vln_bert_optimizer.state_dict(),
                "args": vars(args),
            }, output_path)
            print("Saved pilot adapter: %s" % output_path)

    # main_nav's training loop and environment remain the official DUET code.
    import r2r.main_nav as main_nav
    main_nav.GMapNavAgent = FrozenFusionAgent
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    main_nav.main(args)
    with open(Path(args.output_dir) / 'pilot_resources.json', 'w') as stream:
        json.dump({"wall_seconds": time.perf_counter() - started,
                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "mode": pilot.mode}, stream, indent=2)


if __name__ == "__main__":
    main()
