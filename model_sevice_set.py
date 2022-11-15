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

# model
print('-- model loading ... ...')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.parameters()
print('-- model loaded successfully.')
