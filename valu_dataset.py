import numpy as np
import os
import re
import json
import torch
import torch.optim as optim
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel, AutoModel
from transformers.utils.import_utils import SENTENCEPIECE_IMPORT_ERROR


class Valu_Sent_Comp_Dataset(Dataset):
    def __init__(self, path="",random_idx=int, prefix="train"):
        # assert os.path.isdir(path)
        self.data_path = path
        self.sentence = []
        self.headline = []
        self.random = random_idx
        self.datalist = []
        for n in os.listdir(self.data_path):
          if os.path.splitext(n)[1] == '.json':
            self.datalist.append(n)
        os.getcwd()
        os.chdir(self.data_path)
        for m in range(len(self.datalist)):
            if m == self.random:
              with open(self.datalist[m], encoding="utf-8", mode = 'r') as f:
                context = json.load(f)
                for i in context:
                  element_sent = context[i]['sentence']
                  self.sentence.append(element_sent)
                  element_head = context[i]['headline']
                  self.headline.append(element_head)
            else:
               break
        print('-- Valuation dataset is already downloaded and verified')

    def __len__(self):
      return len(self.sentence)

    def __getitem__(self,idx):
        #data_name = self.data_path.split()                                   
        return {'sentence':self.sentence[idx], 'headline':self.headline[idx]}

def valu_collate_batch(batch):
   sentence_list = []
   headline_list = []
   sentence_ids = []
   headline_ids = []
   for unit in batch:
     sentence_list.append(unit['sentence'])
     headline_list.append(unit['headline'])
   return sentence_list,headline_list
