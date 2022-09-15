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

class Sent_Comp_Dataset(Dataset):
    def __init__(self, path="", prefix="train"):
        self.data_path = path
        self.sentence = []
        self.headline = []
        with open(self.data_path, encoding="utf-8", mode = 'r') as source:
          context = json.load(source)
          for i in context:
            element_sent = context[i]['sentence']
            self.sentence.append(element_sent)

            element_head = context[i]['headline']
            self.headline.append(element_head) 
        print('Files already downloaded and verified')

    def __len__(self):
      return len(self.sentence)
       
    def __getitem__(self,idx):
        #data_name = self.data_path.split()                                   
        return {'sentence':self.sentence[idx], 'headline':self.headline[idx]}

def collate_batch(batch):
   sentence_list = []
   headline_list = []
   for unit in batch:
     sentence_list.append(unit['sentence'])
     headline_list.append(unit['headline'])
   return sentence_list,headline_list
