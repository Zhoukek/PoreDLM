source /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/olmo_workflow/OLMo:/mnt/zzbnew/rnamodel/zhoukexuan/poregpt:${PYTHONPATH:-}


python3 scripts/convert_olmo2_to_hf.py \
   --input_dir "/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13_teacher_model_distill0.1_VQ_64k_lemon/output_150m_ctx1280-lr5e4-vqe_teacher/steps/latest-unsharded"  \
   --output_dir "/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13_teacher_model_distill0.1_VQ_64k_lemon/base"  \
   --tokenizer_json_path "/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13_teacher_model_distill0.1_VQ_64k_lemon/tokenizer-64k.json"  \
