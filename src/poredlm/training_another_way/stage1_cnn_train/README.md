# Stage 1: Continuous CNN

This stage reuses `SignalCNN` from
`training_public/stage1_tokenizer_train/modeling_pore_vq_codec.py` with
`cnn_type: 0`. It uses the 768-channel encoder and stride 5, but never calls
the VQ module or creates a codebook.

For a 6000-sample input chunk, the feature sequence is approximately
`[1200, 768]`. The CNN is trained with noisy-input reconstruction and is
then frozen before feature extraction. There is no discrete token ID in this
stage.

## Train

```bash
bash runs/continuous_cnn/run_train.sh
```

The launcher uses `torchrun`. By default it uses eight visible GPUs; override
both variables together when using another GPU layout:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  bash runs/continuous_cnn/run_train.sh
```

The output directory is configured by `training.output_dir`. Checkpoints are
saved in Hugging Face `save_pretrained` format:

```text
save/continuous_cnn/
├── step_5000/
│   ├── config.json
│   ├── model.safetensors
│   ├── modeling_continuous_cnn.py
│   ├── training_config.yaml
│   └── trainer_state.pt
├── final/
└── latest/
```

`latest` is a copy of the most recent checkpoint and is convenient for the
next step. The model can be loaded with:

```python
from modeling_continuous_cnn import ContinuousSignalCNN

model = ContinuousSignalCNN.from_pretrained("save/continuous_cnn/latest")
```

## Extract features

Pass a checkpoint directory, normally `latest` or `final`, to the extraction
launcher:

```bash
bash runs/continuous_cnn/run_extract_features.sh \
  save/continuous_cnn/latest train /path/to/features/train

bash runs/continuous_cnn/run_extract_features.sh \
  save/continuous_cnn/latest valid /path/to/features/valid
```

The extractor writes continuous feature memmaps and compressed index files,
for example:

```text
features/train/
├── features.npy
├── features.csv.gz
└── metadata.json
```

With multi-GPU extraction, the files are written as rank-specific shards such
as `features_rank00000.npy` and `features_rank00000.csv.gz`. Stage 2 reads
these feature directories directly.

For backward compatibility, `extract_features.py` can still load the old
single-file `.pt` checkpoints.
