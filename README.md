# 🧬PoreDLM: A Diffusion Language Model for Nanopore Signal
PoreDLM 是一种面向 Nanopore 原始电流信号的扩散语言模型式基础模型框架。不同于 PoreGPT 采用自回归 Transformer 对离散 signal token 进行 next-token prediction，PoreDiff 借鉴 Embedded Language Flows 和 Diffusion Language Model 的思想，将原始 Nanopore 电流信号首先编码为 stride-level 的连续 signal embedding，并在连续 embedding 空间中通过 Flow Matching / Diffusion Denoising 学习从扰动表征恢复稳定的生物语义表征，最后再通过 CTC、CRF、k-mer decoder 或 base-token decoder 输出 DNA/RNA 序列及修饰状态。

> PoreDiff 的核心思想是：不再把 Nanopore 信号建模为一个只能从左到右预测下一个 token 的自回归问题，而是将其视为一个连续生物电信号语义空间中的去噪、修复与并行解码问题。

## Stage 1: Tokenizer and Encoder Traing

模型架构：1D-CNN Encoder + VQ + BERT + Decoder


## Stage 2: Diffusion Language Model Training

## Stage 3: Downstream_Task Fine-tuning