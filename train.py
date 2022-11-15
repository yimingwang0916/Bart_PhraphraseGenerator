import transformers
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel, AutoModel
import numpy as np
import os
import re
import torch
import torch.optim as optim
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import json
from transformers.utils.import_utils import SENTENCEPIECE_IMPORT_ERROR
from def_train import model_train
import curve
from curve import learning_curve
from curve import valuation_curve
import lv_curve
from lv_curve import curve
import timer
from timer import Timer
import random

# timer
timer = timer.Timer()
timer.tic()

# input
train_data_path ="/home/yiwang/Datasets"
epoch = 1
train_batch_size = 32
model_save_path = '/home/yiwang/Datasets/model/train_test.pth'
train_max_length = 80
learning_rate = 3e-5
random_idx = random.randint(0,9)
train_loss, valu_loss = model_train(train_data_path,
      epoch, train_batch_size,
      model_save_path,
      train_max_length,
      learning_rate,
      random_idx)
print(len(train_loss))
print(len(valu_loss))
#print(train_loss)
#print(valu_loss)
#curve.learning_curve(train_loss)
#curve.valuation_curve(valu_loss)
lv_curve.curve(train_loss,valu_loss)
print('-- Trained model was saved as',model_save_path)
