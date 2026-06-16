import numpy as np

input_path = '/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage1_tokenizer_apple/train/reference/250F601844011_0_0_0_0_references.npy'
output_path = '/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/S0_HG002_UNMOD-35g/stage2_BERT/03_S0_HG002_UNMOD_35g_model_type_1_cnn_type_0_distill_0.1_8k_vq_apple_split_600/temp/reference/1000_references.npy'

data = np.load(input_path)
data_1000 = data[:1000]
np.save(output_path, data_1000)