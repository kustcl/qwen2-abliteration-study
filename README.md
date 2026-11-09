# Qwen2.5-0.5B Refusal-Direction Ablation: An Exploratory Study

**English** | [简体中文](README.zh-CN.md)

This repository documents a small, first-pass interpretability experiment. It asks whether a refusal-associated activation direction can be estimated from a small local calibration set and attenuated through direct, gradient-free edits to Qwen2.5-0.5B-Instruct weights. It does not include model weights, prompts or responses from the original run, or third-party project source code.

## Findings at a glance

On a 24-item English held-out set sampled from the locally supplied calibration corpus, an English lexical refusal-marker detector fired on 23/24 baseline harmful-request responses and 1/24 responses from the selected edited model. On a separate, self-authored 48-prompt bilingual task set, exact QA accuracy changed from 9/12 to 8/12, instruction accuracy from 6/12 to 5/12, and lexical safety-refusal markers from 12/12 to 1/12. The automatic marker is not a safety judgment. Manual review of the selected model's 12 bilingual risk prompts found 4 clearly inappropriate compliance responses, 3 safety discouragements, 1 response too truncated to judge, and several mixed or ambiguous responses. The experiment therefore demonstrates a behavioral shift, not reliable instruction following, universally removed refusals, or preserved safety.

The sample is small, the test prompts have already informed exploratory analysis, and the English external set is not an independent newly collected benchmark. Confidence intervals are wide. These figures are descriptive results for this run, not a benchmark claim.

## Method

The implementation uses Hugging Face Transformers and PyTorch hooks to capture the final real prompt-token residual activations before and after each Qwen2 block. It estimates layer-wise directions with the difference of means, `d = normalize(mean(h_risk) - mean(h_benign))`, and examines layer-wise separability and bootstrap subsample stability.

For a Qwen2 `nn.Linear` output projection with weight layout `[out, in]`, it applies the orthogonal-projection edit

```text
W' = W - alpha * d * (d^T W)
```

to attention output (`o_proj`) and MLP output (`down_proj`) weights. Edits run in FP32 arithmetic and are cast back to the original dtype. Candidate variants are restored from CPU copies of the original weights before every trial. This modifies selected output maps in the residual stream; it is not a complete intervention that subtracts a direction from every hidden state. No SFT, DPO, optimizer, gradient updates, quantization, or TransformerLens are used.

The design is informed by the single-direction refusal hypothesis, mechanistic interpretability, and public examples of mean-difference activation analysis and orthogonal projection. The two GitHub projects below were read as technical references; their implementations, files, and datasets are not included in this repository and were not copied into its source modules.

## Reproduce

Tested on Windows 11, Python 3.9, an RTX 4060 Laptop with 8 GB VRAM, PyTorch 2.7.1+cu118, and Transformers 4.57.6. This Qwen2-specific pilot requires CUDA. Install a PyTorch build compatible with your CUDA driver, then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

The refusal-direction corpus is not redistributed. Obtain the upstream `harmful.txt` and `harmless.txt` described by the reference project, review their respective licenses/terms, and place copies at `data/upstream/harmful.txt` and `data/upstream/harmless.txt`. The experiment writes sampled prompts and full generations under the ignored `data/` and `results/` directories. Do not commit those local artifacts.

```powershell
python -m unittest discover -s tests -v
python -u run_experiment.py
```

After a run, `results/latest_run.json` points to the run directory. To build and verify the selected edited checkpoint locally:

```powershell
$run = (Get-Content results/latest_run.json | ConvertFrom-Json).path
python -u finalize_model.py --run $run
python -u compare_cli.py --run $run
```

Weights and complete prompt/response traces stay local and are excluded by `.gitignore`.

## References and scope

- Rimsky et al., “Steering Llama 2 via Contrastive Activation Addition” (2023): https://arxiv.org/abs/2312.06681
- Zou et al., “Representation Engineering: A Top-Down Approach to AI Transparency” (2023): https://arxiv.org/abs/2310.01405
- Arditi et al., “Refusal in Language Models Is Mediated by a Single Direction” (2024): https://arxiv.org/abs/2406.11717
- Sumandora, `remove-refusals-with-transformers`: https://github.com/Sumandora/remove-refusals-with-transformers
- FailSpy, `abliterator`: https://github.com/FailSpy/abliterator
- Huihui-AI, Qwen2.5-0.5B-Instruct-abliterated-v3 model card: https://huggingface.co/huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3

This is an independent educational prototype, not an official reproduction of any cited project or model. A training/test split on a tiny convenience sample, lexical refusal matching, and same-model authoring/review leave substantial validity limits. The tuned final-model test set was already used in the exploratory workflow. Direction-label separation is correlational and does not establish a unique causal refusal circuit. Removing safety behavior can increase harmful outputs. Do not deploy the edited model in a user-facing or high-stakes system.

This repository's original code and documentation are MIT-licensed. Model, tokenizer, and any separately obtained dataset remain subject to their respective owners' licenses and terms.
