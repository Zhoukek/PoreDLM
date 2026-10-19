# training_another_way_v1

This directory contains the first implementation of the two-stage continuous-input / VQ-target route.

Both stages now follow the existing `training_another_way` W&B pattern. Set
`wandb.enabled: false` in a run config to disable logging, or use
`WANDB_MODE=offline` for local/offline runs.

## Stage 1

`stage1_vq_train` reuses the public `PoreVQCodec` and its `vector_quantize_pytorch` configuration. It trains:

```text
raw signal -> CNN -> VQ codebook -> decoder -> reconstructed signal
```

The Stage 1 checkpoint contains both the CNN encoder and the VQ codebook. The initial config uses an EMA-updated codebook, matching the public tokenizer route.

Run from the repository root, or use the stage launcher:

```bash
USE_NOHUP=0 bash src/poredlm/training_another_way_v1/stage1_vq_train/runs/continuous_vq/run_train.sh
```

## Stage 2

`stage2_continuous_target` freezes the Stage 1 checkpoint and computes:

```text
continuous CNN latent z -> mask -> Transformer -> predicted embedding
                         \
                          -> frozen VQ -> target embedding q
```

The first version deliberately keeps the existing `SmoothL1 + cosine` objective. It does not use token IDs or contrastive learning yet.

After setting `stage1.checkpoint` in
`stage2_continuous_target/runs/continuous_bert_vq_target/config.yaml`, run:

```bash
USE_NOHUP=0 bash src/poredlm/training_another_way_v1/stage2_continuous_target/runs/continuous_bert_vq_target/run_train.sh
```

The Stage 2 target is detached. Therefore Stage 2 cannot change the Stage 1 codebook in this first controlled experiment.
