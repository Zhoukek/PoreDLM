# Stage 2: continuous input with VQ embedding targets

This stage freezes the Stage 1 VQ checkpoint and trains:

```text
continuous CNN latent z -> mask -> Transformer -> predicted embedding
                         \
                          -> frozen VQ -> target embedding q
```

The first experiment uses the existing `SmoothL1 + cosine` target loss. It
does not use token IDs or contrastive learning.

Set `stage1.checkpoint` in `runs/continuous_bert_vq_target/config.yaml`, then
run:

```bash
USE_NOHUP=0 bash runs/continuous_bert_vq_target/run_train.sh
```

