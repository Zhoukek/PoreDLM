# Architecture

PoreDLM is organized around three stages:

1. Stage 1 trains a raw-signal tokenizer and encoder using a 1D-CNN, VQ module,
   contextual encoder, and reconstruction or sequence-oriented decoder.
2. Stage 2 trains a diffusion language model over continuous stride-level signal
   embeddings using denoising or flow matching objectives.
3. Stage 3 fine-tunes the pretrained representation for downstream tasks such as
   basecalling and modification detection.

This document is a placeholder for model diagrams, tensor shapes, and training
objectives.
