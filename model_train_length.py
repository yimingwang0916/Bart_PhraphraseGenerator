import transformers
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel, AutoModel
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
import training_length_control
from training_length_control import model_train_length
import Plots_creater
from Plots_creater import LL_plot,RL_plot

# timer
timer = model_set.Timer()
timer.tic()

# input
train_data_path ="/home/yiwang/Datasets/train"
epoch = 1
train_batch_size = 32
model_save_path = '/home/yiwang/Datasets/model/10_32_80_3e5_slides.pth'
train_max_length = 80
learning_rate = 3e-5
random_idx = random.randint(0,9)
train_loss, source_len, output_len, reference_len, ratio_OS,ratio_RS = training_length_control.model_train_length(train_data_path,
      epoch, train_batch_size,
      model_save_path,
      train_max_length,
      learning_rate,
      )

Plots_creater.LL_plot(source_len, output_len, reference_len)
Plots_creater.RL_plot(ratio_OS,ratio_RS)
