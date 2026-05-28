# 🧬PoreDLM: A Diffusion Language Model for Nanopore Signal
PoreDLM 是一种面向 Nanopore 原始电流信号的扩散语言模型式基础模型框架。不同于 PoreGPT 采用自回归 Transformer 对离散 signal token 进行 next-token prediction，PoreDiff 借鉴 Embedded Language Flows 和 Diffusion Language Model 的思想，将原始 Nanopore 电流信号首先编码为 stride-level 的连续 signal embedding，并在连续 embedding 空间中通过 Flow Matching / Diffusion Denoising 学习从扰动表征恢复稳定的生物语义表征，最后再通过 CTC、CRF、k-mer decoder 或 base-token decoder 输出 DNA/RNA 序列及修饰状态。

> PoreDiff 的核心思想是：不再把 Nanopore 信号建模为一个只能从左到右预测下一个 token 的自回归问题，而是将其视为一个连续生物电信号语义空间中的去噪、修复与并行解码问题。

## Stage 1: Tokenizer Traing

> 模型架构：1D-CNN Encoder + VQ + Decoder

bash PoreDLM/src/poredlm/training/stage1_tokenizer/runs/test_zhou/run.sh

~~~
数据准备：
1. 
~~~

## Stage 2: BERT_Encoder Traing

~~~
数据准备：
1. 



~~~


## Stage 3: Diffusion Language Model Training

## Stage 4: Downstream_Task Fine-tuning

## Project Structure

```text
PoreDLM/
├── configs/                 # YAML configs for the three training stages
├── data/                    # Local data workspace; large files are ignored by Git
├── docs/                    # Architecture notes and experiment documentation
├── scripts/                 # CLI entry points for training and preprocessing
├── src/poredlm/             # Python package source code
│   ├── data/                # Datasets and preprocessing utilities
│   ├── models/              # Encoder, diffusion backbone, and decoders
│   ├── tasks/               # Downstream task definitions
│   ├── tokenization/        # VQ/tokenizer and signal embedding modules
│   ├── training/            # Stage-specific training loops
│   └── utils/               # Shared config and runtime helpers
└── tests/                   # Unit tests and smoke tests
```

要求：项目是支持多卡的环境，并且保证可重复性实验

## Multi-GPU and Reproducibility

PoreDLM is designed for distributed training and reproducible experiments.

Recommended launch pattern:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_stage1_tokenizer.py --config configs/stage1_tokenizer.yaml
torchrun --standalone --nproc_per_node=4 scripts/train_stage2_diffusion.py --config configs/stage2_diffusion.yaml
torchrun --standalone --nproc_per_node=4 scripts/train_stage3_finetune.py --config configs/stage3_finetune.yaml
```

Each stage config includes:

```yaml
reproducibility:
  seed: 42
  deterministic: true
  benchmark: false
  warn_only: true

distributed:
  enabled: true
  backend: nccl
  find_unused_parameters: false
```
