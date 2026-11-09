# -*- coding: utf-8 -*-
"""统一中文 CLI：每条输入先显示原始模型，再显示越狱版模型。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import torch
from transformers import TextStreamer
from src.modeling import Runner, seed_all

ROOT = Path(__file__).resolve().parent
LABELS = {"strong": "越狱版模型（Abliteration）", "weak": "越狱版模型（Abliteration，弱消融）"}
FOLDERS = {"strong": "final_model", "weak": "exported_model"}


def resolve_run(value=None):
    if value:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    return Path(json.loads((ROOT / "results/latest_run.json").read_text(encoding="utf-8"))["path"])


@torch.inference_mode()
def answer(runner, prompt, max_new_tokens, temperature, seed, stream=True):
    seed_all(seed)
    inputs = runner.encode([prompt])
    generation = deepcopy(runner.generation_config)
    generation.max_new_tokens = max_new_tokens
    generation.do_sample = temperature > 0
    if temperature > 0:
        generation.temperature, generation.top_p, generation.top_k = temperature, 0.9, 50
    streamer = TextStreamer(runner.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
    t0 = time.perf_counter()
    output = runner.model.generate(**inputs, generation_config=generation, use_model_defaults=False, streamer=streamer)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tokens = output[0, inputs.input_ids.shape[1]:].tolist()
    eos = generation.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    end = next((i for i, t in enumerate(tokens) if t in eos), len(tokens))
    response = runner.tokenizer.decode(tokens[:end], skip_special_tokens=True)
    if not stream:
        print(response, flush=True)
    print(f"[生成 {end} token，用时 {elapsed:.2f} 秒" + ("，达到长度上限]" if end == max_new_tokens else "]"), flush=True)
    return {"response": response, "generated_tokens": end, "seconds": elapsed, "truncated": end == max_new_tokens}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="原始模型 / 越狱版模型（Abliteration）对比")
    parser.add_argument("--run", help="实验目录，默认最近一组")
    parser.add_argument("--edited", choices=list(FOLDERS), default="strong", help="strong 越狱版；weak 弱消融")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="每个模型最大生成长度，默认 256")
    parser.add_argument("--temperature", type=float, default=0, help="0 为贪心，正数为采样；两边重置相同种子")
    parser.add_argument("--prompt", help="单条测试，输出两边回答后退出")
    parser.add_argument("--prompt-file", help="从 UTF-8 文本读取单条 prompt，支持多行")
    parser.add_argument("--save-json", help="可选：保存本次对比结果到指定 JSON 文件")
    parser.add_argument("--no-stream", action="store_true", help="生成完毕后一次性打印")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 2048 or not 0 <= args.temperature <= 2:
        parser.error("生成长度需为 1~2048；温度需为 0~2")
    if args.prompt and args.prompt_file:
        parser.error("--prompt 与 --prompt-file 只能指定一个")
    run = resolve_run(args.run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config["max_input_tokens"] = 4096
    seed_all(config["seed"])
    original = (ROOT / config["model_path"]).resolve()
    print("Qwen2.5-0.5B｜原始模型 / 越狱版模型（Abliteration）", flush=True)
    print("正在加载原始模型……", flush=True)
    baseline = Runner(original, config)
    edited = None
    active = args.edited

    def load_edited(kind):
        nonlocal edited, active
        target = run / FOLDERS[kind]
        if not (target / "model.safetensors").exists():
            script = "finalize_model.py" if kind == "strong" else "export_variant.py"
            raise FileNotFoundError(f"未找到模型：{target}；请先运行 {script}。")
        if edited is not None:
            del edited
            edited = None
            torch.cuda.empty_cache()
        print(f"正在加载{LABELS[kind]}……", flush=True)
        edited = Runner(target, config)
        active = kind

    load_edited(active)
    print("加载完成。输入问题，依次显示原始模型和越狱版模型的回答。", flush=True)
    print("命令：/quit 退出；/strong 越狱版；/weak 弱消融；/tokens 512 设置长度；/help 帮助。", flush=True)
    print("每条问题独立比较，不沿用上条对话。", flush=True)
    supplied = Path(args.prompt_file).read_text(encoding="utf-8-sig") if args.prompt_file else args.prompt
    records = []
    while True:
        try:
            prompt = supplied if supplied is not None else input("\n你 > ")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        prompt = prompt.strip()
        if not prompt:
            if supplied is not None:
                parser.error("prompt 不能为空")
            continue
        if supplied is None and prompt.startswith("/"):
            if prompt in {"/quit", "/exit", "/q"}:
                break
            if prompt in {"/strong", "/weak"}:
                load_edited(prompt[1:]); continue
            if prompt.startswith("/tokens "):
                try:
                    count = int(prompt.split()[1])
                    if not 1 <= count <= 2048:
                        raise ValueError()
                    args.max_new_tokens = count
                    print(f"两边生成长度上限已设为 {count}。")
                except (ValueError, IndexError):
                    print("用法：/tokens 512（范围 1~2048）")
                continue
            print("/strong：越狱版；/weak：弱消融；/tokens 数字：长度；/quit：退出。")
            continue
        if supplied is not None:
            print(f"\n你 > {prompt}", flush=True)
        record = {"prompt": prompt, "edited_variant": active, "max_new_tokens": args.max_new_tokens,
                  "temperature": args.temperature, "seed": config["seed"]}
        try:
            print("\n──────── 原始模型 ────────", flush=True)
            record["original"] = answer(baseline, prompt, args.max_new_tokens, args.temperature, config["seed"], not args.no_stream)
            print(f"\n──────── {LABELS[active]} ────────", flush=True)
            record["edited"] = answer(edited, prompt, args.max_new_tokens, args.temperature, config["seed"], not args.no_stream)
        except ValueError as exc:
            print(f"输入无法处理：{exc}。请缩短问题。", flush=True)
            if supplied is not None:
                raise
            continue
        records.append(record)
        if args.save_json:
            target = Path(args.save_json)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        if supplied is not None:
            break


if __name__ == "__main__":
    main()
