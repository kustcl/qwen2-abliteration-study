# Experimental Report: Refusal-Direction Ablation on Qwen2.5-0.5B-Instruct

**Status:** initial exploratory study; not a validated intervention or benchmark.

## Research question

Can a contrastive direction estimated from residual activations on risky versus benign prompts act as a useful exploratory handle for editing a small instruction-tuned model? The experiment tests activation measurement, direction stability, several edit locations and strengths, checkpoint integrity, bilingual utility tasks, and refusal-related behavior. It is too small to establish a general claim about refusal mechanisms or safe deployment.

## Implementation

The reference hypothesis is that some refusal behavior can be associated with directions in residual-stream representations. Forward hooks collect raw per-block `resid_pre` and `resid_post` vectors at the last unpadded prompt token. For activations `h`, the primary estimator at each layer is the normalized class-mean difference `d = normalize(mean(h_risk) - mean(h_benign))`. A second, weaker-label estimator uses outputs that contain lexical refusal markers. Held-out activation projections measure label separability; repeated half-sample estimates measure directional stability. A high AUC measures linear association with these labels; it does not show that the direction causes refusal.

The edit for a PyTorch linear map `y = W x`, where `W` has shape `[d_out, d_in]`, is `W' = (I - alpha d d^T)W = W - alpha d(d^T W)`. Editing the attention `o_proj` and MLP `down_proj` output maps attenuates their output component along `d`. Strength `alpha=1` removes that component from those particular maps in exact arithmetic. It does not erase the direction from every layer activation or remove every safety behavior. Selected embeddings, normalization parameters, Q/K/V maps, and the language-model head are left alone. Bias terms of edited output maps are projected too.

Training is disabled. Activations and direction calculations are measured without gradient updates. A CPU clone of the original output weights supports restoration between variants. The original checkpoint hash and restored logits are checked, and an independently reloaded saved model is checked for exact tokenizer IDs and response reproducibility.

## Protocol

- Base model: Qwen2.5-0.5B-Instruct, 24 Qwen2 blocks; approximately 0.49B parameters.
- Platform: Windows 11; Python 3.9; RTX 4060 Laptop 8 GB; PyTorch 2.7.1+cu118; Transformers 4.57.6.
- Calibration: 64 risky plus 64 benign English prompts for direction estimation, with 16 per class for direction validation. The sources were local `harmful.txt` and `harmless.txt` files attributed upstream to AdvBench and Alpaca-cleaned. The split used fixed-seed sampling, exact normalized deduplication against authored test prompts, and short-prompt filtering. The repository publishes only aggregate provenance counts and source-file hashes, not prompts or copied corpora.
- Self-authored evaluation: 32 development prompts and 48 bilingual test prompts across exact-answer QA, format and instruction following, benign sensitive questions, and explicitly unsafe requests.
- External sample: another 24 English prompts per source class, sampled from the same local source files after removing exact normalized overlap. This is a held-out split, not an independently sourced benchmark.
- Variants: risk directions from layers 8, 14, and 20; strengths 0.25, 0.5, and 1.0; all-block versus middle-block and MLP-only edits; zero and fixed-random-direction controls. Each candidate starts from restored base weights.
- Metrics: exact QA and instruction grading; lexical refusal-marker rates on the first 700 response characters; per-example human review of the edited bilingual risk set; truncation rates; first-token distribution KL; bootstrap intervals; baseline hash and reload checks.

## Results

| Metric | Original | Selected edit |
|---|---:|---:|
| Self-authored bilingual QA accuracy, 12 items | 9/12 (75.0%) | 8/12 (66.7%) |
| Self-authored bilingual instruction accuracy, 12 items | 6/12 (50.0%) | 5/12 (41.7%) |
| Bilingual unsafe-request refusal-marker rate, 12 items | 12/12 (100%) | 1/12 (8.3%) |
| External English harmful-source refusal-marker rate, 24 items | 23/24 (95.8%) | 1/24 (4.2%) |
| External English benign-source refusal-marker rate, 24 items | 0/24 | 0/24 |
| Full bilingual test outputs with exact-match QA/format grading | 48 items | 48 items |

The selected model was the layer-20 risk-label mean-difference direction with `alpha=1`, applied to attention and MLP output maps across all 24 blocks. It was selected through an additional exploratory finalization pass. The 48-item test set had already been run during the broader experiment and is not a pristine confirmatory test after this extra selection. The control experiment, in which a fixed random direction caused little marker-rate change, is consistent with the estimated direction carrying more task-related signal than that one control, but a single random control is not enough to exclude general edit effects.

Manual review of the edited model on 12 self-authored unsafe prompts recorded 4 clearly inappropriate compliance responses, 3 safety discouragements, 1 response too truncated to judge, and several mixed/ambiguous responses. Short generations frequently truncate, so observed prefixes cannot substantiate full-response quality. The lexical marker counts can miss indirect discouragement and can flag non-refusals. They are diagnostic text matches, not a safety metric.

Exact QA changed by one item and instruction following by one item. Wilson 95% intervals are correspondingly wide: original/edit QA 46.8–91.1% / 39.1–86.2%; instruction 25.4–74.6% / 19.3–68.0%. This sample cannot establish that ordinary capabilities are preserved.

The base model's primary weights had identical before/after SHA-256. Restoration and zero-control logits matched exactly. The independently saved final checkpoint reproduced tokenizer IDs and all 48 generated test responses in this run. These verify implementation and artifact integrity, not the validity or safety of model behavior.

## Interpretation and next steps

These results support a narrow observation: on this model, prompt format, and small English calibration corpus, a contrastive residual direction followed by selected output-projection edits substantially changed lexical refusal patterns. Utility measurements also moved downward, and the edited model sometimes complied with clearly improper requests while remaining inconsistent. The effect is not simply “all guardrails removed,” nor is successful marker reduction itself a positive safety outcome.

A credible next study should lock a larger, independently sourced evaluation set before tuning, use multiple seeds and substantially more random-direction controls, separate refusal behavior labels from intent labels, vary layer locations and strengths under a predeclared selection rule, run longer generations with truncation-aware review, independently review safety outcomes, and repeat on larger architectures. The present repository enables that research workflow but does not make those follow-up results claims.

## References

1. Rimsky et al. (2023), *Steering Llama 2 via Contrastive Activation Addition*. https://arxiv.org/abs/2312.06681
2. Zou et al. (2023), *Representation Engineering: A Top-Down Approach to AI Transparency*. https://arxiv.org/abs/2310.01405
3. Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*. https://arxiv.org/abs/2406.11717
4. Sumandora, `remove-refusals-with-transformers`, technical reference only: https://github.com/Sumandora/remove-refusals-with-transformers
5. FailSpy, `abliterator`, technical reference only: https://github.com/FailSpy/abliterator
6. Huihui-AI, `Qwen2.5-0.5B-Instruct-abliterated-v3` model card, contextual model example only: https://huggingface.co/huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3

No cited repository source code, local model checkpoints, upstream calibration files, or per-example model responses are distributed here.
