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
import training_set
from training_set import model_train
import Plots_creater
from Plots_creater import LV_plot


# timer
timer = model_set.Timer()
timer.tic()

# input
train_data_path ="/home/yiwang/Datasets/train"
epoch = 8
train_batch_size = 32
model_save_path = '/home/yiwang/Datasets/model/10_32_80_3e5_slides.pth'
train_max_length = 80
learning_rate = 3e-5
train_loss, valu_loss = training_set.model_train(train_data_path,
      epoch, train_batch_size,
      model_save_path,
      train_max_length,
      learning_rate,
      )

Plots_creater.LV_plot(train_loss,valu_loss)
print('-- Trained model was saved as',model_save_path)
