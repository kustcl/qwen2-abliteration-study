"""Qwen2 native Transformers activation collection and reversible weight edits."""
from contextlib import contextmanager
import hashlib
import json
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Runner:
    def __init__(self, model_path, config):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; this experiment requires the configured GPU.")
        self.config = config
        self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False, fix_mistral_regex=False,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            dtype=getattr(torch, config["dtype"]), attn_implementation="sdpa",
        ).to(self.device).eval().requires_grad_(False)
        if self.model.config.model_type != "qwen2":
            raise ValueError("This implementation is audited for Qwen2 only.")
        self.layers = self.model.model.layers
        self.generation_config = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=config["max_new_tokens"],
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
        )
        # Transformers >=4.50 can fill global defaults from a re-saved model.
        # Persist the exact experiment generation configuration on all exports.
        self.model.generation_config = self.generation_config

    def prompt(self, text):
        return self.tokenizer.apply_chat_template([
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": text},
        ], tokenize=False, add_generation_prompt=True)

    def encode(self, prompts):
        encoded = self.tokenizer([self.prompt(p) for p in prompts], return_tensors="pt", padding=True, add_special_tokens=False)
        if encoded.input_ids.shape[1] > self.config["max_input_tokens"]:
            raise ValueError("Input exceeds configured maximum; refusing silent truncation.")
        return encoded.to(self.device)

    @torch.inference_mode()
    def generate(self, rows):
        results = []
        size = self.config["batch_size"]
        eos = self.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for start in range(0, len(rows), size):
            batch = rows[start:start + size]
            inputs = self.encode([r["prompt"] for r in batch])
            t0 = time.perf_counter()
            outputs = self.model.generate(**inputs, generation_config=self.generation_config, use_model_defaults=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tokens = outputs[:, inputs.input_ids.shape[1]:].cpu().tolist()
            for row, tok in zip(batch, tokens):
                end = next((i for i, t in enumerate(tok) if t in eos), len(tok))
                results.append(dict(row, response=self.tokenizer.decode(tok[:end], skip_special_tokens=True),
                                    generated_tokens=end, truncated=end == self.config["max_new_tokens"],
                                    batch_seconds=elapsed))
        return results

    @torch.inference_mode()
    def activations(self, rows):
        """Raw pre/post block residuals at final prompt token; no final RMSNorm confusion."""
        collected = {"pre": [], "post": []}
        cache = {}
        handles = []
        def make_pre(i):
            def hook(module, args):
                cache[("pre", i)] = args[0][:, -1, :].float().cpu()
            return hook
        def make_post(i):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                cache[("post", i)] = hidden[:, -1, :].float().cpu()
            return hook
        try:
            for i, layer in enumerate(self.layers):
                handles.append(layer.register_forward_pre_hook(make_pre(i)))
                handles.append(layer.register_forward_hook(make_post(i)))
            for start in range(0, len(rows), self.config["batch_size"]):
                inputs = self.encode([r["prompt"] for r in rows[start:start + self.config["batch_size"]]])
                self.model.model(**inputs, use_cache=False)
                for point in collected:
                    collected[point].append(torch.stack([cache[(point, i)] for i in range(len(self.layers))], dim=1))
        finally:
            for handle in handles:
                handle.remove()
        return {key: torch.cat(values) for key, values in collected.items()}

    @torch.inference_mode()
    def response_activations(self, rows, token_count=8):
        """Diagnostic response-prefix activations, not used to select edit directions.

        Teacher-force each original response's first 8 tokens; mean raw resid_pre
        over those positions. Lexical leakage makes this a descriptive diagnostic.
        """
        result = []
        cache = {}
        handles = []
        def hook_for(i):
            def hook(module, args):
                cache[i] = args[0][:, -response_length:, :].float().mean(dim=1).cpu()
            return hook
        try:
            for i, layer in enumerate(self.layers):
                handles.append(layer.register_forward_pre_hook(hook_for(i)))
            for row in rows:
                prefix = self.tokenizer.encode(self.prompt(row["prompt"]), add_special_tokens=False)
                answer = self.tokenizer.encode(row["response"], add_special_tokens=False)[:token_count]
                if not answer:
                    answer = [self.tokenizer.eos_token_id]
                response_length = len(answer)
                ids = torch.tensor([prefix + answer], device=self.device)
                self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                result.append(torch.stack([cache[i] for i in range(len(self.layers))], dim=1))
        finally:
            for handle in handles:
                handle.remove()
        return torch.cat(result)

    @torch.inference_mode()
    def first_token_logits(self, rows):
        outputs = []
        for start in range(0, len(rows), self.config["batch_size"]):
            inputs = self.encode([r["prompt"] for r in rows[start:start + self.config["batch_size"]]])
            hidden = self.model.model(**inputs, use_cache=False).last_hidden_state[:, -1]
            outputs.append(self.model.lm_head(hidden).float().cpu())
        return torch.cat(outputs)


def output_modules(model, layers, target="both"):
    if target not in {"both", "attn", "mlp"}:
        raise ValueError("Unknown projection target")
    for i in layers:
        if not 0 <= i < len(model.model.layers):
            raise ValueError("Invalid edit layer")
        layer = model.model.layers[i]
        if target in {"both", "attn"}:
            yield f"model.layers.{i}.self_attn.o_proj", layer.self_attn.o_proj
        if target in {"both", "mlp"}:
            yield f"model.layers.{i}.mlp.down_proj", layer.mlp.down_proj


@torch.no_grad()
def project_weight(weight, direction, strength):
    """Linear weight is [output,input]: W'=(I-alpha*d*d^T)W, in FP32."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must lie in [0,1]")
    d = direction.float().to(weight.device)
    if d.ndim != 1 or d.numel() != weight.shape[0] or not torch.isfinite(d).all() or d.norm() < 1e-8:
        raise ValueError("Invalid output-space direction")
    d = d / d.norm()
    w = weight.float()
    edited = w - strength * torch.outer(d, d @ w)
    return edited.to(weight.dtype)


class WeightEditor:
    """CPU clones, never state_dict aliases. Every variant starts from originals."""
    def __init__(self, runner):
        self.runner = runner
        self.originals = {}
        for name, module in output_modules(runner.model, range(len(runner.layers))):
            self.originals[name + ".weight"] = module.weight.detach().cpu().clone()
            if module.bias is not None:
                self.originals[name + ".bias"] = module.bias.detach().cpu().clone()

    @torch.no_grad()
    def restore(self):
        for name, module in output_modules(self.runner.model, range(len(self.runner.layers))):
            module.weight.copy_(self.originals[name + ".weight"])
            if module.bias is not None:
                module.bias.copy_(self.originals[name + ".bias"])

    @torch.no_grad()
    def apply(self, direction, spec):
        self.restore()
        stats = []
        alpha = spec["strength"]
        for name, module in output_modules(self.runner.model, spec["edit_layers"], spec.get("target", "both")):
            before = module.weight.float()
            d = direction.float().to(before.device)
            d = d / d.norm()
            projected_before = (d @ before).norm()
            module.weight.copy_(project_weight(module.weight, d, alpha))
            after = module.weight.float()
            stats.append({"name": name, "relative_weight_change": ((after - before).norm() / before.norm()).item(),
                          "direction_component_ratio": ((d @ after).norm() / projected_before.clamp_min(1e-12)).item()})
            if module.bias is not None:
                b = module.bias.float()
                module.bias.copy_((b - alpha * d * (d @ b)).to(module.bias.dtype))
        return stats

    @contextmanager
    def temporary(self, direction, spec):
        try:
            stats = self.apply(direction, spec)
            yield stats
        finally:
            self.restore()


def next_token_kl(reference, candidate):
    """KL(original || edited) on full-vocabulary first response-token distribution."""
    p = reference.double().log_softmax(-1)
    q = candidate.double().log_softmax(-1)
    return (p.exp() * (p - q)).sum(-1).mean().item()
