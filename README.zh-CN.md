# Qwen2.5-0.5B 拒答方向消融：初步探索

[English](README.md) | **简体中文**

本项目记录一次小模型、初次尝试的机制可解释性实验：从小规模本地校准数据估计与拒答相关的激活方向，再通过无梯度权重编辑减弱它对 Qwen2.5-0.5B-Instruct 输出的影响。仓库不包含模型权重、原实验的完整 prompt/回答，也不包含第三方参考项目的源码或文件。

## 初步结果

在来自本地校准语料的 24 条英文留出请求中，基于英文拒答词表的自动检测器在原模型 23/24 个风险请求回答中触发，在选定编辑模型中触发 1/24 次。在另一个自建的 48 条中英双语任务集上，严格问答正确率由 9/12 变为 8/12，指令遵循率由 6/12 变为 5/12，风险请求的拒答词表触发率由 12/12 变为 1/12。关键词指标不等于安全判断。逐条人工复核编辑模型的 12 个双语风险请求，发现 4 条明显不当配合、3 条安全劝阻、1 条回答因截断无法判断，其余表现混杂或有歧义。因此，这组结果说明模型行为发生了变化，不足以证明拒答都已消失、指令遵循可靠或安全能力得到保留。

样本量很小，探索中已经查看过这些测试提示；外部英文集也不是新采集的独立基准。统计区间较宽。这些数值仅描述本次运行，不代表普遍效果或基准排名。

## 技术路线

实现使用 Hugging Face Transformers 与 PyTorch forward hooks，采集 Qwen2 每个 block 前后最后一个有效提示 token 的 residual activation。采用均值差估计逐层方向：`d = normalize(mean(h_risk) - mean(h_benign))`，并检查各层的标签可分性和子样本重采样稳定性。

对权重布局为 `[out, in]` 的 Qwen2 `nn.Linear` 输出投影，使用正交方向投影：

```text
W' = W - alpha * d * (d^T W)
```

将其用于注意力输出矩阵 `o_proj` 和 MLP 输出矩阵 `down_proj`。投影以 FP32 计算后转回原精度；每个实验候选都从 CPU 保存的原始权重副本恢复，再独立编辑。此方法修改若干残差流输出映射，不等价于在每个隐藏状态上完整减去同一方向。实验不做 SFT、DPO、梯度训练或量化，也不使用 TransformerLens。

路线参考单方向拒答假说、表示工程与公开的均值差激活分析和正交投影示例。下方两个 GitHub 项目仅作为阅读和理解实现思路的技术参考；它们的代码、文件和数据集均未放入本仓库，也未复制到实验模块中。

## 复现实验

已在 Windows 11、Python 3.9、RTX 4060 Laptop 8 GB 显存、PyTorch 2.7.1+cu118 和 Transformers 4.57.6 下运行。本试验专门适配 Qwen2 模型架构，需要 CUDA。先安装与你的驱动兼容的 PyTorch CUDA 版本，再运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

本项目不再分发拒答方向校准语料。请根据参考项目说明，从上游获取 `harmful.txt` 与 `harmless.txt`，自行确认对应授权条款后，分别放到 `data/upstream/harmful.txt` 和 `data/upstream/harmless.txt`。程序会将抽样 prompt 和生成回答写入已忽略的本地 `data/` 与 `results/` 目录。不要提交这些本地实验数据。

```powershell
python -m unittest discover -s tests -v
python -u run_experiment.py
```

实验结束后，可通过 `results/latest_run.json` 找到当前实验目录，在本地生成并验证编辑模型：

```powershell
$run = (Get-Content results/latest_run.json | ConvertFrom-Json).path
python -u finalize_model.py --run $run
python -u compare_cli.py --run $run
```

模型权重和完整 prompt/回答保留在本地，并由 `.gitignore` 排除。

## 技术来源与边界

- Rimsky 等，《Steering Llama 2 via Contrastive Activation Addition》(2023)：https://arxiv.org/abs/2312.06681
- Zou 等，《Representation Engineering: A Top-Down Approach to AI Transparency》(2023)：https://arxiv.org/abs/2310.01405
- Arditi 等，《Refusal in Language Models Is Mediated by a Single Direction》(2024)：https://arxiv.org/abs/2406.11717
- Sumandora，`remove-refusals-with-transformers`：https://github.com/Sumandora/remove-refusals-with-transformers
- FailSpy，`abliterator`：https://github.com/FailSpy/abliterator
- Huihui-AI，Qwen2.5-0.5B-Instruct-abliterated-v3 模型卡：https://huggingface.co/huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3

这是独立的教学与方法探索原型，不是任何引用项目或模型的官方复现。小样本、词表检测器以及由同一研究者构建和复核的测例都会限制结论有效性；最终模型的测试集已参与本次探索。方向标签的可分性是相关性证据，不能证明存在唯一且因果的拒答回路。削弱安全行为可能增加有害输出风险。请勿将编辑模型部署到面向用户或高风险场景。

本仓库原创代码与文档采用 MIT 协议。模型、分词器及自行另行取得的数据集仍适用其各自所有者的授权和使用条款。
