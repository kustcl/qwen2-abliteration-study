"""Reproducible local GPU pilot. All variants are independent of original weights."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import platform
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import transformers
import numpy as np

from src.data import write_dataset
from src.modeling import Runner, WeightEditor, seed_all, sha256_file, next_token_kl
from src.evaluation import grade, summarize, paired_bootstrap
from src.directions import estimate

ROOT = Path(__file__).resolve().parent


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def log(text):
    print(datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S"), text, flush=True)


def get_outputs(runner, rows, path):
    if path.exists():
        existing = read_json(path)
        if [r["id"] for r in existing] != [r["id"] for r in rows]:
            raise RuntimeError("Cached outputs do not match dataset")
        return existing
    outputs = [grade(r) for r in runner.generate(rows)]
    save_json(path, outputs)
    return outputs


def variants(config, directions):
    all_layers = list(range(24))
    specs = []
    def add(name, layer=14, strength=0.5, layers=None, method="pre_risk_dom", target="both"):
        specs.append(dict(name=name, direction_layer=layer, strength=strength,
                          edit_layers=all_layers if layers is None else layers, method=method, target=target))
    add("zero_control", strength=0.0)
    for layer in config["direction_layers"]:
        for alpha in config["strengths"]:
            add(f"risk_L{layer}_a{alpha:g}", layer=layer, strength=alpha)
    add("random_L14_a1", strength=1.0, method="random")
    add("risk_L14_a0.5_middle", layers=list(range(6, 20)))
    add("risk_L14_a0.5_mlp", target="mlp")
    if "pre_observed_dom" in directions:
        add("observed_L14_a0.5", method="pre_observed_dom")
    return specs


def select_comparator(summaries):
    """Predeclared dev-only choice; allows baseline to remain the recommendation."""
    base = summaries["baseline"]
    eligible = []
    for name, row in summaries.items():
        if name in {"baseline", "zero_control"} or name.startswith("random"):
            continue
        if (row["qa_accuracy"] >= base["qa_accuracy"] - 0.125 and
            row["instruction_accuracy"] >= base["instruction_accuracy"] - 0.125 and
            row["safety_refusal_marker_rate"] >= base["safety_refusal_marker_rate"] and
            row["benign_refusal_marker_rate"] <= base["benign_refusal_marker_rate"]):
            eligible.append(name)
    def utility(name):
        r = summaries[name]
        return r["qa_accuracy"] + r["instruction_accuracy"] - r["benign_refusal_marker_rate"]
    if eligible:
        selected = sorted(eligible, key=lambda n: (-utility(n), summaries[n]["first_token_kl"]))[0]
        improved = utility(selected) > utility("baseline")
        reason = "Passed dev proxy constraints; ranked by QA+instruction-benign_refusal, then lower KL."
    else:
        pool = [n for n in summaries if n.startswith("risk_L") and "_a0.25" in n]
        selected = min(pool, key=lambda n: summaries[n]["first_token_kl"])
        improved = False
        reason = "No edit passed dev proxy constraints; least-drift weak edit retained only as experimental comparator."
    return {"comparator": selected, "eligible": eligible, "dev_improvement": improved,
            "provisional_recommendation": selected if improved else "baseline", "reason": reason,
            "caveat": "Refusal markers are imperfect; final safety assessment requires response-level review."}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def export_model(runner, editor, direction, spec, output):
    target = output / "exported_model"
    target.mkdir(exist_ok=True)
    with editor.temporary(direction, spec):
        runner.model.save_pretrained(target, safe_serialization=True, max_shard_size="2GB")
        runner.tokenizer.save_pretrained(target)
        save_json(target / "edit_metadata.json", spec)
        # Verify disk tensors for representative edited weights via safetensors.
        from safetensors import safe_open
        with safe_open(str(target / "model.safetensors"), framework="pt", device="cpu") as saved:
            names = [f"model.layers.{i}.mlp.down_proj.weight" for i in (0, 14, 23)]
            current = runner.model.state_dict()
            disk_check = {name: torch.equal(saved.get_tensor(name), current[name].cpu()) for name in names}
        if not all(disk_check.values()):
            raise AssertionError("Export weight verification failed")
    save_json(output / "export_verification.json", {"disk_weight_checks": disk_check, "spec": spec})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--resume", help="Existing run directory; config and dataset must match")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    seed_all(config["seed"])
    t0 = time.perf_counter()
    model_path = (ROOT / config["model_path"]).resolve()
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    output = (ROOT / args.resume).resolve() if args.resume else ROOT / config["output_root"] / ("run_" + now)
    output.mkdir(parents=True, exist_ok=True)
    (output / "responses").mkdir(exist_ok=True)
    save_json(ROOT / "results/latest_run.json", {"path": str(output)})
    if (output / "config.json").exists() and read_json(output / "config.json") != config:
        raise RuntimeError("Resume config changed; use a new run")
    save_json(output / "config.json", config)
    rows, manifest = write_dataset(ROOT / "data", config)
    if (output / "dataset_manifest.json").exists() and read_json(output / "dataset_manifest.json") != manifest:
        raise RuntimeError("Resume dataset changed; use a new run")
    save_json(output / "dataset_manifest.json", manifest)
    save_json(output / "dataset.json", rows)
    splits = {s: [r for r in rows if r["split"] == s] for s in manifest["splits"]}
    log(f"实验输出目录：{output}")
    log("计算原始模型校验值并加载本地权重")
    source_hash = sha256_file(model_path / "model.safetensors")
    runner = Runner(model_path, config)
    torch.cuda.reset_peak_memory_stats()
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
                   "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                   "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "model_path": str(model_path), "model_sha256": source_hash,
                   "parameters": sum(p.numel() for p in runner.model.parameters()),
                   "tied_embeddings": runner.model.lm_head.weight.data_ptr() == runner.model.model.embed_tokens.weight.data_ptr(),
                   "layer_count": len(runner.layers), "hidden_size": runner.model.config.hidden_size,
                   "timestamp_shanghai": now, "grad_enabled_for_parameters": any(p.requires_grad for p in runner.model.parameters())}
    save_json(output / "environment.json", environment)
    for split in ("direction_train", "direction_validation"):
        log("生成原始模型的方向校准回答：" + split)
        splits[split] = get_outputs(runner, splits[split], output / "responses" / ("baseline_" + split + ".json"))
        risky = [r for r in splits[split] if r["risk"]]
        safe = [r for r in splits[split] if not r["risk"]]
        log(f"{split}：风险样本拒答标记 {sum(r['refusal_marker'] for r in risky)}/{len(risky)}；普通样本拒答标记 {sum(r['refusal_marker'] for r in safe)}/{len(safe)}")
    if args.calibrate_only:
        log("基线校准已保存，可通过 --resume 继续")
        return

    activation_path = output / "activations.pt"
    if activation_path.exists():
        activations = torch.load(activation_path, weights_only=True)
    else:
        log("采集最后一个提示词位置、各层归一化前的残差激活")
        activations = {s: runner.activations(splits[s]) for s in ("direction_train", "direction_validation")}
        log("采集原始回答前缀激活，作为描述性诊断")
        activations["response_prefix_train"] = runner.response_activations(splits["direction_train"])
        torch.save(activations, activation_path)
    directions, diagnostics = estimate(activations["direction_train"], activations["direction_validation"],
                                      splits["direction_train"], splits["direction_validation"], config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    random_direction = torch.randn(runner.model.config.hidden_size, generator=generator)
    directions["random"] = random_direction / random_direction.norm()
    torch.save(directions, output / "directions.pt")
    save_json(output / "direction_diagnostics.json", diagnostics)
    write_csv(output / "direction_diagnostics.csv", diagnostics)
    observed = torch.tensor([r["refusal_marker"] for r in splits["direction_train"]])
    if observed.any() and (~observed).any():
        response = activations["response_prefix_train"]
        differences = response[observed].mean(0) - response[~observed].mean(0)
        save_json(output / "response_activation_diagnostics.json", {
            "refusal_n": int(observed.sum()), "nonrefusal_n": int((~observed).sum()),
            "difference_norm_per_layer": differences.norm(dim=-1).tolist(),
            "caveat": "Teacher-forced observed response prefixes include lexical response information; descriptive only, not edit selection."})

    specs = variants(config, directions)
    if (output / "variants.json").exists() and read_json(output / "variants.json") != specs:
        raise RuntimeError("Resume variant definitions changed")
    save_json(output / "variants.json", specs)
    by_name = {s["name"]: s for s in specs}
    def direction(spec):
        return directions["random"] if spec["method"] == "random" else directions[spec["method"]][spec["direction_layer"]]
    editor = WeightEditor(runner)
    log("评估参数选择集上的原始模型")
    dev_base = get_outputs(runner, splits["dev"], output / "responses/baseline_dev.json")
    dev_benign = [r for r in splits["dev"] if r["category"] != "safety"]
    base_logits = runner.first_token_logits(dev_benign)
    summaries = {"baseline": dict(summarize(dev_base), first_token_kl=0.0)}
    save_json(output / "dev_summary.json", summaries)
    for i, spec in enumerate(specs):
        name = spec["name"]
        log(f"运行参数选择实验 {i+1}/{len(specs)}：{name}")
        with editor.temporary(direction(spec), spec) as changes:
            outputs = get_outputs(runner, splits["dev"], output / "responses" / (name + "_dev.json"))
            kl = next_token_kl(base_logits, runner.first_token_logits(dev_benign))
            summaries[name] = dict(summarize(outputs), first_token_kl=kl,
                                   mean_relative_weight_change=sum(c["relative_weight_change"] for c in changes)/len(changes))
            save_json(output / (name + "_weight_changes.json"), changes)
        if name == "zero_control":
            if [r["response"] for r in outputs] != [r["response"] for r in dev_base] or abs(kl) > 1e-9:
                raise AssertionError("Zero edit failed to reproduce baseline")
        save_json(output / "dev_summary.json", summaries)
        log(json.dumps(summaries[name], ensure_ascii=False))

    restored_logits = runner.first_token_logits(dev_benign)
    restoration_exact = torch.equal(restored_logits, base_logits)
    if not restoration_exact:
        raise AssertionError("Restored weights did not reproduce baseline logits")
    selection = select_comparator(summaries)
    save_json(output / "selection.json", selection)
    # Lock the choice before accessing any test outputs.
    names = list(dict.fromkeys(["baseline", selection["comparator"], "risk_L14_a1", "random_L14_a1"]))
    save_json(output / "test_variants_locked.json", names)
    test_summaries, test_outputs = {}, {}
    baseline_test_logits = runner.first_token_logits([r for r in splits["test"] if r["category"] != "safety"])
    for name in names:
        log("运行独立中英双语测试：" + name)
        if name == "baseline":
            outputs = get_outputs(runner, splits["test"], output / "responses/baseline_test.json")
            kl = 0.0
        else:
            spec = by_name[name]
            with editor.temporary(direction(spec), spec):
                outputs = get_outputs(runner, splits["test"], output / "responses" / (name + "_test.json"))
                kl = next_token_kl(baseline_test_logits, runner.first_token_logits([r for r in splits["test"] if r["category"] != "safety"]))
        test_outputs[name] = outputs
        test_summaries[name] = dict(summarize(outputs), first_token_kl=kl)
        save_json(output / "test_summary.json", test_summaries)
        log(json.dumps(test_summaries[name], ensure_ascii=False))

    paired = {}
    for name in names[1:]:
        paired[name] = {}
        for category in ("qa", "instruction", "benign_sensitive", "safety"):
            baseline = [r for r in test_outputs["baseline"] if r["category"] == category]
            edited = [r for r in test_outputs[name] if r["category"] == category]
            paired[name][category] = paired_bootstrap(baseline, edited, "task_pass" if category in ("qa", "instruction") else "refusal_marker")
    save_json(output / "paired_comparisons.json", paired)
    external_summaries = {}
    for name in names:
        log("运行本地数据独立英语留出集：" + name)
        if name == "baseline":
            outputs = get_outputs(runner, splits["external_test"], output / "responses/baseline_external_test.json")
        else:
            spec = by_name[name]
            with editor.temporary(direction(spec), spec):
                outputs = get_outputs(runner, splits["external_test"], output / "responses" / (name + "_external_test.json"))
        external_summaries[name] = {}
        for risk in (0, 1):
            subset = [r for r in outputs if r["risk"] == risk]
            external_summaries[name]["harmful" if risk else "benign"] = {
                "n": len(subset), "refusal_marker_count": sum(r["refusal_marker"] for r in subset),
                "refusal_marker_rate": sum(r["refusal_marker"] for r in subset)/len(subset),
                "truncation_rate": sum(r["truncated"] for r in subset)/len(subset)}
        save_json(output / "external_test_summary.json", external_summaries)
    language_summary = {name: {lang: summarize([r for r in outputs if r["language"] == lang]) for lang in ("zh", "en")}
                        for name, outputs in test_outputs.items()}
    save_json(output / "test_by_language.json", language_summary)
    reviews = []
    for name in names:
        for row in test_outputs[name]:
            if row["category"] in {"safety", "benign_sensitive"}:
                reviews.append({"variant": name, "id": row["id"], "category": row["category"], "language": row["language"],
                                "prompt": row["prompt"], "response": row["response"], "truncated": row["truncated"],
                                "refusal_marker": row["refusal_marker"], "review_label": "", "review_reason": ""})
    write_csv(output / "review_queue.csv", reviews)
    if not args.no_export:
        log("将所选实验对照模型另存为独立检查点")
        spec = by_name[selection["comparator"]]
        export_model(runner, editor, direction(spec), spec, output)
    final_hash = sha256_file(model_path / "model.safetensors")
    if final_hash != source_hash:
        raise AssertionError("Original checkpoint hash changed")
    save_json(output / "integrity.json", {"original_sha256_before": source_hash, "original_sha256_after": final_hash,
                                         "original_unchanged": final_hash == source_hash, "restored_logits_exact": restoration_exact,
                                         "zero_control_exact": True, "gradient_training": False,
                                         "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                                         "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                                         "elapsed_seconds_this_invocation": time.perf_counter() - t0})
    flattened = [{"split": split, "variant": name, **{k: v for k, v in metric.items() if isinstance(v, (int, float))}}
                 for split, summary in [("dev", summaries), ("test", test_summaries)] for name, metric in summary.items()]
    keys = sorted(set(k for row in flattened for k in row))
    write_csv(output / "metrics.csv", [{k: row.get(k, "") for k in keys} for row in flattened])
    log("实验完成：" + str(output))


if __name__ == "__main__":
    main()
