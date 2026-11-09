"""独立重载导出模型，验证其生成结果与内存编辑模型一致。"""
import argparse
import json
from pathlib import Path
import sys
from src.modeling import Runner, seed_all


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="核对导出模型能否独立重载并复现回答")
    parser.add_argument("--run", required=True)
    parser.add_argument("--variant", help="默认验证开发集所选弱编辑模型")
    args = parser.parse_args()
    path = Path(args.run).resolve()
    load = lambda name: json.loads((path / name).read_text(encoding="utf-8"))
    config, selection = load("config.json"), load("selection.json")
    seed_all(config["seed"])
    variant = args.variant or selection["comparator"]
    model_dir = "abliterated_model" if variant == "risk_L14_a1" else "exported_model"
    runner = Runner(path / model_dir, config)
    expected = load(f"responses/{variant}_test.json")[:config["batch_size"]]
    actual = runner.generate(expected)
    matches = [{"id": old["id"], "response_exact": old["response"] == new["response"]}
               for old, new in zip(expected, actual)]
    results = {"reloaded_locally": True, "samples": matches, "all_exact": all(r["response_exact"] for r in matches)}
    (path / f"{model_dir}_reload_verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if not results["all_exact"]:
        raise AssertionError("导出模型重载后的回答未完全一致")
    print(f"导出模型独立重载成功，{len(matches)} 条回答完全一致。")


if __name__ == "__main__":
    main()
