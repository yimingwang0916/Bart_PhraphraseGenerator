import transformers_len.src.transformers.models.bart.modeling_bart
from transformers_len.src.transformers.models.bart.modeling_bart import BartForConditionalGeneration, BartModel
import transformers_len.src.transformers.models.bart.tokenization_bart
from transformers_len.src.transformers.models.bart.tokenization_bart import BartTokenizer
import numpy as np
import os
import re
import random
import torch
import torch.optim as optim
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import json
from transformers.utils.import_utils import SENTENCEPIECE_IMPORT_ERROR

import model_set
from model_set import Timer
import training_set
from training_set import model_train
import Plots_creater
from Plots_creater import LV_plot


# timer
timer = model_set.Timer()
timer.tic()

# input
train_data_path = "/home/yiwang/Datasets/Quora/normal/train"
valu_data_path = "/home/yiwang/Datasets/Quora/normal/valuation"
epoch = 10
train_batch_size = 64
model_save_path = '/home/yiwang/Datasets/model/LA/Quora_8_64_80_e-5.pth'
train_max_length = 80
learning_rate = 1e-5
#, valu_loss, source_len, output_len, reference_len, ratio_OS,ratio_RS
train_loss = training_set.model_train(train_data_path,
      valu_data_path,
      epoch,
      train_batch_size,
      model_save_path,
      train_max_length,
      learning_rate,
      )

train_loss_array = np.array(train_loss)
np.save('/home/yiwang/Datasets/figure_config/LA/Quora/normal/8_64_80_e-5/train_loss.npy',train_loss_array)
#valu_loss_array = np.array(valu_loss)
#np.save('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_3e-5/valu_loss.npy',valu_loss_array)

#source_len = np.array(source_len)
#np.save('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_3e-5/source_len.npy',source_len)
#output_len = np.array(output_len)
#np.save('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_3e-5/output_len.npy',output_len)
#reference_len = np.array(reference_len)
#np.save('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_3e-5/reference_len.npy',reference_len)
#ratio_OS = np.array(ratio_OS)
#np.save('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_3e-5/ratio_OS.npy',ratio_OS)
#ratio_RS = np.array(ratio_RS)
#np.save('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_3e-5/ratio_RS.npy',ratio_RS)

print('-- Trained model was saved as',model_save_path)
