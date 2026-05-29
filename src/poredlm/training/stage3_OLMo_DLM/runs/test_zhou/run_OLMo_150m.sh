#!/bin/bash
# nohup ./run.sh > train.log 2>&1 &
# 参考: https://github.com/allenai/OLMo/blob/bf536fdfb5ab9b77c8defac2d7ca37db05eea733/scripts/augusta/peteish1-anneal.sh

# 如何断点续训: 将如下行替换下面命令中的--load_path="" 
# --load_path="/workspace/zzb_tutorial/olmo20m_pt_output/steps" 
#export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca259235j( j forfor)
# export WANDB_API_KEY=PasteYourWandbApikeyHere
source /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/poregpt
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=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

torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:29502 \
	scripts/train.py configs/config_20m_vqed-1k_ctx1280_dna32g_teacher.yaml \
	--run_name="olmo-pt-bioseq-20m-dna32g-split1280_overlap256-vqe_teacher-1k" \
        --wandb.entity="zhoukek-zhejiang-university" \
        --wandb.project="olmo-pt" \
        --load_path="" \
        --save_folder="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13_teacher_model_distill0.1_VQ_1k_lemon/output_20m_ctx1280-lr5e4-vqe_teacher/steps/" 2>&1 | tee run.log
