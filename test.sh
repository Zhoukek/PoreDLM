#!/bin/bash

set -e

ref=/mnt/zzbnew/rnamodel/data/ref/HG002.fasta
model=/mnt/zzbnew/rnamodel/model/bonito/dna_basic_0121

fast5_root=/mnt/zzbnew/rnamodel/wangxue/data/DNA_data/S0_HG002_UNMOD/250F601844011/fast5_split_one

out_root=/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/basecall_chunk

device=cuda:0

mkdir -p "$out_root"

# 手动指定要跑的大文件夹
batch_list=(
    250F601844011_0_0_0_0
)

for batch_id in "${batch_list[@]}"; do
    batch_dir="$fast5_root/$batch_id"

    if [ ! -d "$batch_dir" ]; then
        echo "[WARN] Batch dir not found, skip: $batch_dir"
        continue
    fi

    echo "######################################"
    echo "[INFO] Processing batch: $batch_id"
    echo "[INFO] Batch dir: $batch_dir"
    echo "######################################"

    for fast5_dir in "$batch_dir"/*; do
        [ -d "$fast5_dir" ] || continue

        part_id=$(basename "$fast5_dir")

        outdir="$out_root/$batch_id/$part_id"
        outfile="$outdir/acc95.bam"
        logfile="$outdir/basecall.log"

        mkdir -p "$outdir"

        if [ -s "$outfile" ]; then
            echo "[SKIP] $batch_id/$part_id already exists: $outfile"
            continue
        fi

        echo "======================================"
        echo "[INFO] Basecalling: $batch_id/$part_id"
        echo "[INFO] Input : $fast5_dir"
        echo "[INFO] Output: $outfile"
        echo "======================================"

        bonito basecaller "${model}" "${fast5_dir}" \
            --no-trim \
            --reference "${ref}" \
            --save-ctc \
            --min-accuracy-save-ctc 0.95 \
            --batchsize 128 \
            --device "${device}" \
            --chunksize 6000 \
            --overlap 3000 \
            > "${outfile}" 2> "${logfile}"
            
        echo "[DONE] $batch_id/$part_id"
    done
done

echo "[ALL DONE]"