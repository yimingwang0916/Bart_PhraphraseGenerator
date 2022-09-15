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

# inputs
test_data_path ='/content/gdrive/My Drive/sentence_compression/short_data/eval_test.json'
test_batch_size = 8
model_save_path = '/content/gdrive/My Drive/Model/short_test.pth'
test_max_length = 80

test(test_data_path, 
     test_batch_size, 
     model_save_path, 
     test_max_length
     )
