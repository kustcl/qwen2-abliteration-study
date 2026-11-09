"""从原始权重与已记录方向独立重建两种可直接加载的编辑模型。"""
import argparse
import json
import shutil
from pathlib import Path
import sys
import torch
from src.modeling import Runner, WeightEditor, seed_all, sha256_file

ROOT = Path(__file__).resolve().parent


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="独立导出本次强消融和弱编辑模型")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    path = Path(args.run).resolve()
    load = lambda name: json.loads((path / name).read_text(encoding="utf-8"))
    config, selection = load("config.json"), load("selection.json")
    seed_all(config["seed"])
    source = (ROOT / config["model_path"]).resolve()
    original_hash = sha256_file(source / "model.safetensors")
    runner = Runner(source, config)
    editor = WeightEditor(runner)
    directions = torch.load(path / "directions.pt", weights_only=True)
    specs = {s["name"]: s for s in load("variants.json")}
    exported = {}
    for name, folder in [(selection["comparator"], "exported_model"), ("risk_L14_a1", "abliterated_model")]:
        spec = specs[name]
        target = path / folder
        target.mkdir(exist_ok=True)
        with editor.temporary(directions[spec["method"]][spec["direction_layer"]], spec):
            runner.model.save_pretrained(target, safe_serialization=True, max_shard_size="2GB")
            runner.tokenizer.save_pretrained(target)
            if (source / "LICENSE").exists():
                shutil.copyfile(source / "LICENSE", target / "LICENSE")
            # Keep the tokenizer architecture explicit in exported metadata.
            metadata = json.loads((target / "tokenizer_config.json").read_text(encoding="utf-8"))
            metadata["model_type"] = "qwen2"
            (target / "tokenizer_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            (target / "edit_metadata.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        exported[folder] = {"variant": name, "sha256": sha256_file(target / "model.safetensors"),
                            "generation_config": runner.generation_config.to_dict()}
        print(f"模型已保存：{target}", flush=True)
    if sha256_file(source / "model.safetensors") != original_hash:
        raise AssertionError("原始模型校验失败")
    (path / "standalone_exports.json").write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
