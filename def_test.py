mport model_device_set
import dataset
from dataset import Sent_Comp_Dataset
from dataset import collate_batch
import valu_dataset
from valu_dataset import Valu_Sent_Comp_Dataset
from valu_dataset import valu_collate_batch

import numpy as np
import json
import os
import re
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor
from torch.utils.data import Dataset, DataLoader
# transformers
import transformers
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel, AutoModel

from transformers.utils.import_utils import SENTENCEPIECE_IMPORT_ERROR
import nltk
from nltk.translate.bleu_score import sentence_bleu

def model_test(test_data_path=str, test_batch_size = int, model_save_path = str, test_max_length = int):

   model_device_set.model.load_state_dict(torch.load(model_save_path)) # loads only the model parameters
   model_device_set.model.eval()

   scd_test = Sent_Comp_Dataset(test_data_path)
   test_dataloader = DataLoader(scd_test, batch_size=test_batch_size, shuffle=True, collate_fn=collate_batch, drop_last=False)

   nums = []

   for batch_idx,(sentences,headlines) in enumerate(test_dataloader):

      sent_ids = model_device_set.tokenizer.batch_encode_plus(sentences, max_length=test_max_length, return_tensors="pt", pad_to_max_length=True)
     # head_ids = model_device_set.tokenizer.batch_encode_plus(headlines, max_length=test_max_length, return_tensors="pt", pad_to_max_length=True)
     # y = head_ids['input_ids']

      summaries = model_device_set.model.generate(
       input_ids = sent_ids["input_ids"].to(model_device_set.device),
       attention_mask = sent_ids["attention_mask"].to(model_device_set.device),
       num_beams=4,
       length_penalty=2.0,
       max_length=142,  # +2 from original because we start at step=1 and stop before max_length
       min_length=0,  # +1 from original because we start at step=1
       no_repeat_ngram_size=3,
       early_stopping=True,
       do_sample=False,
      )

      model_predictions = model_device_set.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
      for j in range(len(sentences)):
        print('-- evaluated',j,'pair of',batch_idx+1,'batch is following:')
        print('-- source sentence is:')
        print(sentences[j])
        print('-- reference sentence is:')
        print(headlines[j])
        reference_head_ids = model_device_set.tokenizer(headlines[j])
        reference_tokens = reference_head_ids['input_ids']
        str1_reference = []
        for k in range(len(reference_tokens)):
          str1 = str(reference_tokens[k])
          str1_reference.append(str1)
          reference = [str1_reference]

        print('-- generated sentence is:')
        print(model_predictions[j])
        candidate_head_ids = model_device_set.tokenizer(model_predictions[j])
        candidate_tokens = candidate_head_ids['input_ids']
        str2_candidate = []
        for l in range(len(candidate_tokens)):
          str2 = str(candidate_tokens[l])
          str2_candidate.append(str2)
          candidate = str2_candidate

        score = sentence_bleu(reference, candidate)
        print(score)
        nums.append(score)

   np.mean(nums)
   print('-- current BLEU score:',np.mean)

   return nums
  # dec = [model_device_set.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
  # print('--',dec)

