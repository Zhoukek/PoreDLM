# Stage 4 basecall 实验

这个目录保存基于 Stage 3 self-flow DLM 的 basecall 训练配置。下面说明如何用训练得到的最佳 checkpoint 对 JSONL reads 做 basecall，并输出 FASTQ。

## 模型和代码路径

本次推理使用：

```text
Checkpoint:
src/poredlm/training_public/stage4_basecall_for_RNA/experiment_20206_10_22/HF_RSQ741C12A529_DNADLLM_V013_ode_hidden_4_layer_tcn_ctc_300000_chunks/runs/HF_RSQ741C12A529_DNADLLM_V013_ode_hidden_4_layer_tcn_ctc_360000_chunks/ckpt_best.pt

Stage 3 DLM:
src/poredlm/training_public/stage3_DLM_train_new/scripts/runs/HF_RSQ741C12A529_DNABERT_V554_flow_matching/hf_dlm_step10000_new

推理入口:
src/poredlm/training_public/stage4_basecall_for_RNA/Basecalling/basecaller_v8_0420/infer.py
```

`ckpt_best.pt` 只保存 Stage 4 basecall 模型参数；加载时仍需提供训练时使用的 Stage 3 DLM 目录。训练参数可在对应实验的 [`run.sh`](HF_RSQ741C12A529_DNADLLM_V013_ode_hidden_4_layer_tcn_ctc_300000_chunks/run.sh) 查看。

## 输入格式

推理入口接受普通 `.jsonl` 或 `.jsonl.gz` 文件。每行是一个 JSON 对象，至少有 read ID 和包含 Bwav token 的 `text` 字段：

```json
{"read_id":"read_001","text":"<|bwav:12|><|bwav:98|><|bwav:37|>"}
```

read ID 字段也可以叫 `id`。没有 `read_id`/`id` 或没有 `text` 的行会被跳过。输入应是和训练数据相同格式的 Bwav token 序列，不是 FAST5/POD5 原始信号文件。

训练使用 `bwav` tokenizer，并在 tokenizer 内部加 `128` 的词表偏移。因此通常不要额外传 `--token_offset 128`；推理参数保持 `--token_offset 0`，避免把偏移重复加到输入 token 上。

## `context_hidden`、`ode_hidden` 和 `sde_hidden`

这三个选项决定 Stage 4 basecall head 从 DLM 的哪一阶段取特征。它们的张量维度可以相同，但数值和含义不同：

| 特征 | 特征来源 | 是否经过 flow denoiser | 随机性 |
| --- | --- | --- | --- |
| `context_hidden` | DLM context encoder 的输出表示 | 否 | 无 |
| `ode_hidden` | 以 context 表示为起点，经过 flow denoiser 做 ODE 积分后的表示 | 是 | 无，参数相同时结果确定 |
| `sde_hidden` | 以 context 表示为起点，积分时加入随机噪声并经过 flow denoiser 修正后的表示 | 是 | 有；可用 `--sde_seed` 固定随机序列 |

`--ode_steps` / `--sde_steps` 是积分步数，`--ode_start_t` / `--sde_start_t` 是起始时间；`--sde_gamma` 控制 SDE 路径的噪声强度，`--sde_self_cond_cfg_scale` 和 `--ode_self_cond_cfg_scale` 控制 denoiser 的 self-conditioning CFG。只有选了相应的 `feature_source`，对应参数才生效。`context_hidden` 不执行这段 ODE/SDE 积分。

**basecall head 是按训练时的特征类型训练的。** 当前 `ckpt_best.pt` 使用 `ode_hidden`，所以推荐先用下方 ODE 配置作为基线。把它改成 `context_hidden` 或 `sde_hidden`，或更改 ODE/SDE 参数，都会改变送入 head 的特征；这可用于对照实验，但不能假设准确率不变。要让某种特征成为正式推理配置，最好用同一种特征及参数训练或微调对应的 head，并在验证集上比较。

## 运行 basecall

在项目根目录执行。先按机器情况修改 `CUDA_VISIBLE_DEVICES` 和输入/输出路径：

```bash
cd /mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM
source src/poredlm/training/set_env.sh

export CUDA_VISIBLE_DEVICES=0
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$PWD/src/poredlm/training_public/stage4_basecall_for_RNA:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

CKPT="$PWD/src/poredlm/training_public/stage4_basecall_for_RNA/experiment_20206_10_22/HF_RSQ741C12A529_DNADLLM_V013_ode_hidden_4_layer_tcn_ctc_300000_chunks/runs/HF_RSQ741C12A529_DNADLLM_V013_ode_hidden_4_layer_tcn_ctc_360000_chunks/ckpt_best.pt"
BASE_MODEL="$PWD/src/poredlm/training_public/stage3_DLM_train_new/scripts/runs/HF_RSQ741C12A529_DNABERT_V554_flow_matching/hf_dlm_step10000_new"
INPUT_JSONL_GZ="/path/to/input.jsonl.gz"
OUTPUT_FASTQ="$PWD/basecall_output.fastq"

python -m Basecalling.basecaller_v8_0420.infer \
  --ckpt "$CKPT" \
  --model_name_or_path "$BASE_MODEL" \
  --jsonl_gz "$INPUT_JSONL_GZ" \
  --out "$OUTPUT_FASTQ" \
  --device cuda --amp \
  --decoder auto \
  --add_cls_token --cls_token_id 5 \
  --feature_source ode_hidden \
  --hidden_layer -1 \
  --ode_steps 1 \
  --ode_start_t 0.97 \
  --ode_self_cond_cfg_scale 0.5 \
  --pre_head_type auto \
  --pre_head_smoothing none \
  --head_output_activation tanh \
  --head_output_scale 5 \
  --backbone_chunk_size 1536 \
  --token_offset 0 \
  --batch_size 4 \
  --max_tokens 2048 \
  --overlap 128
```

运行结束后，basecall 序列写入 `OUTPUT_FASTQ`。如果显存不足，可调低 `--batch_size`；较长 read 会按 `--max_tokens` 分块并通过 `--overlap` 拼接。输出目录会由推理程序自动创建。

### 参数要和训练保持一致

此 checkpoint 用 `ode_hidden` 特征训练，训练时 ODE 配置为 `1` 步、`start_t=0.97`、CFG scale `0.5`；训练还启用了 CLS token（ID `5`）、TCN pre-head、`tanh` 输出激活、输出缩放 `5` 和 backbone chunk size `1536`。上面的命令已按这些设置恢复推理模型。若改成 `context_hidden`、不同 ODE 参数或不同 head 参数，输入给 basecall head 的特征会改变，结果不再是训练配置下的基线推理。

## 用 SDE 特征做对照（可选）

推理代码也支持 `--feature_source sde_hidden`。运行时把上面命令中的 `--feature_source ode_hidden` 和 ODE 参数替换为：

```bash
--feature_source sde_hidden \
--sde_steps 4 \
--sde_start_t 0.85 \
--sde_gamma 0.1 \
--sde_self_cond_cfg_scale 1.0 \
--sde_seed 6198
```

这是用 ODE 特征训练的 head 接收 SDE 特征的实验对照，特征分布与训练时不同，应和上面的 ODE 基线分别跑验证集并比较。固定 `--sde_seed` 可复现实验；不传 seed 则使用运行时随机数状态。
