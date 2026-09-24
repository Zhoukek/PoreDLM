# Stage 1: continuous CNN

This stage reuses `SignalCNN` from
`training_public/stage1_tokenizer_train/modeling_pore_vq_codec.py` with
`cnn_type: 0`. It uses the 768-channel encoder and stride 5, but never calls
the VQ module or creates a codebook.

For a 6000-sample input chunk, the feature sequence is approximately
`[1200, 768]`. After training, extract frozen features with:

```bash
bash runs/continuous_cnn/run_extract_features.sh save/continuous_cnn/last.pt train /path/to/features/train
bash runs/continuous_cnn/run_extract_features.sh save/continuous_cnn/last.pt valid /path/to/features/valid
```
