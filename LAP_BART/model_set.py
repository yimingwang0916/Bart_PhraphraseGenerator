import transformers_len.src.transformers.models.bart.modeling_bart
from transformers_len.src.transformers.models.bart.modeling_bart import BartForConditionalGeneration
import transformers_len.src.transformers.models.bart.tokenization_bart 
from transformers_len.src.transformers.models.bart.tokenization_bart import BartTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import time

## model
print('-- model loading ... ...')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

## device
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model.to(device)
model.parameters()
print('-- model loaded succesfully.')

## training dataset
class Sent_Comp_Dataset(Dataset):
    def __init__(self, path="", prefix="train"):
        # assert os.path.isdir(path)
        self.data_path = path
        self.sentence = []
        self.headline = []
        self.datalist = []
        for n in os.listdir(self.data_path):
          if os.path.splitext(n)[1] == '.json':
            self.datalist.append(n)
        print(self.datalist)
        os.getcwd()
        os.chdir(self.data_path)
        for m in range(len(self.datalist)):
              with open(self.datalist[m], encoding="utf-8", mode = 'r') as f:
                  context = json.load(f)
                  for i in context:
                    element_sent = context[i]['sentence']
                    self.sentence.append(element_sent)
                    element_head = context[i]['headline']
                    self.headline.append(element_head)
        print('-- datasets are already downloaded and verified')

    def __len__(self):
      return len(self.sentence)

    def __getitem__(self,idx):
        #data_name = self.data_path.split()                                   
        return {'sentence':self.sentence[idx], 'headline':self.headline[idx]}

def collate_batch(batch):
   sentence_list = []
   headline_list = []
   sentence_ids = []
   headline_ids = []
   for unit in batch:
     sentence_list.append(unit['sentence'])
     headline_list.append(unit['headline'])
   return sentence_list,headline_list

## valuation dataset
class Quora_Sent_Comp_Dataset(Dataset):
    def __init__(self, path="", prefix="train"):
        # assert os.path.isdir(path)
        self.data_path = path
        self.sentence = []
        self.headline = []
        self.datalist = []
        for n in os.listdir(self.data_path):
          if os.path.splitext(n)[1] == '.json':
            self.datalist.append(n)
        os.getcwd()
        os.chdir(self.data_path)
        for m in range(len(self.datalist)):
              with open(self.datalist[m], encoding="utf-8", mode = 'r') as f:
                context = json.load(f)
                for i in context:
                  element_sent = context[i]['question_1']
                  element_head = context[i]['question_2']
                  self.sentence.append(element_sent)
                  self.sentence.append(element_head)
                  self.headline.append(element_head)
                  self.headline.append(element_sent)
         
        print('-- Quora dataset is already downloaded and verified')

    def __len__(self):
      return len(self.sentence)

    def __getitem__(self,idx):
        #data_name = self.data_path.split()                                   
        return {'sentence':self.sentence[idx], 'headline':self.headline[idx]}

def Quora_collate_batch(batch):
   sentence_list = []
   headline_list = []
   sentence_ids = []
   headline_ids = []
   for unit in batch:
     sentence_list.append(unit['sentence'])
     headline_list.append(unit['headline'])
   return sentence_list,headline_list

## tmier  
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.warm_up = 0
        self.gmtime = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()
        self.gmtime = time.gmtime()
        print('-- Start time is',self.gmtime)
        return(self.start_time)

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        if self.warm_up < 10:
            self.warm_up += 1
            print('-- Time spent is',self.diff)
            return self.diff
        else:
            self.total_time += self.diff
            self.calls += 1
            self.average_time = self.total_time / self.calls

        if average:
            print('-- Time spent is',self.average_time)
            return self.average_time

        else:
            print('-- Time spent is',self.diff)
            return self.diff
