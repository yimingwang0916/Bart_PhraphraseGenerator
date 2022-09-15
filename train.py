import numpy as np 
import os
import re
import torch
import torch.optim as optim
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import json
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel, AutoModel
from transformers.utils.import_utils import SENTENCEPIECE_IMPORT_ERROR

import dataset
from dataset import Sent_Comp_Dataset, collate_batch
import def_train_test
from def_train_test import train

# model 
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)
model.parameters()

# input
train_data_path ='/content/gdrive/My Drive/sentence_compression/preprocessed_data'
epoch = 1
train_batch_size = 8
model_save_path = '/content/gdrive/My Drive/Model/short_train.pth'
train_max_length = 80

train(train_data_path, 
      epoch, train_batch_size, 
      model_save_path, 
      train_max_length)
