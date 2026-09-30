# Stage 1: VQ signal autoencoder

This stage reuses the public `PoreVQCodec` and trains:

```text
raw signal -> CNN encoder -> VQ codebook -> decoder -> reconstructed signal
```

The experiment launcher and configuration live under `runs/continuous_vq/`.

```bash
USE_NOHUP=0 bash runs/continuous_vq/run_train.sh
```

The resulting `latest` checkpoint contains the CNN encoder and VQ codebook,
which are consumed by Stage 2.

