import model_set
from model_set import Sent_Comp_Dataset, collate_batch, Quora_Sent_Comp_Dataset, Quora_collate_batch
import numpy as np
import json
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel, AutoModel
from transformers.utils.import_utils import SENTENCEPIECE_IMPORT_ERROR
import nltk
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score
from evaluate import load

def dataloader(test_data_path = str, test_batch_size = int):
    scd_test = Quora_Sent_Comp_Dataset(test_data_path)
    test_dataloader = DataLoader(scd_test, batch_size=test_batch_size, shuffle=True, collate_fn=Quora_collate_batch, drop_last=False)
    return test_dataloader

def length_obtain(sentences = str, model_predictions = str, headlines = str,source_len=list, output_len=list, reference_len=list):

    sentences_len = sentences.split()
    source_len.append(len(sentences_len))
    model_predictions_len = model_predictions.split()
    output_len.append(len(model_predictions_len))
    headlines_len = headlines.split()
    reference_len.append(len(headlines_len))  

    return source_len,output_len,reference_len 

## trained_model_BLEU
def trained_model_BLEU(test_data_path=str, test_batch_size = int, model_save_path = str, test_max_length = int):

   model_set.model.load_state_dict(torch.load(model_save_path)) # loads only the model parameters
   model_set.model.eval()

   test_dataloader = dataloader(test_data_path,test_batch_size)   
   nums = []
   source_len = []
   output_len = []
   reference_len = []

   for batch_idx,(sentences,headlines) in enumerate(test_dataloader):

      sent_ids = model_set.tokenizer.batch_encode_plus(sentences, max_length=test_max_length, return_tensors="pt", pad_to_max_length=True)
     # head_ids = model_device_set.tokenizer.batch_encode_plus(headlines, max_length=test_max_length, return_tensors="pt", pad_to_max_length=True)
     # y = head_ids['input_ids']

      summaries = model_set.model.generate(
       input_ids = sent_ids["input_ids"].to(model_set.device),
       attention_mask = sent_ids["attention_mask"].to(model_set.device),
       num_beams=4,
       length_penalty=2.0,
       max_length=142,  # +2 from original because we start at step=1 and stop before max_length
       min_length=0,  # +1 from original because we start at step=1
       no_repeat_ngram_size=3,
       early_stopping=True,
       do_sample=False,
      )

      model_predictions = model_set.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
      for j in range(len(sentences)):
        print('-- evaluated',j,'pair of',batch_idx+1,'batch is following:')
        print('-- source sentence is:')
        print(sentences[j])
        print('-- reference sentence is:')
        print(headlines[j])
        reference = [headlines[j]]
        print('-- generated sentence is:')
        print(model_predictions[j])
        candidate = model_predictions[j]
        score = sentence_bleu(reference, candidate, weights = (1,))
        print(score)
        nums.append(score)
        source_len, output_len, reference_len = length_obtain(sentences[j], model_predictions[j], headlines[j], source_len, output_len, reference_len)
    
   np_mean = np.mean(nums)
   print('-- current bert score:',np_mean)

   return nums,np_mean,source_len,output_len,reference_len

## trained_model_bert
def trained_model_bert(test_data_path=str, test_batch_size = int, model_save_path = str, test_max_length = int):

   model_set.model.load_state_dict(torch.load(model_save_path)) # loads only the model parameters
   model_set.model.eval()

   test_dataloader = dataloader(test_data_path,test_batch_size)
   nums = []

   for batch_idx,(sentences,headlines) in enumerate(test_dataloader):

      sent_ids = model_set.tokenizer.batch_encode_plus(sentences, max_length=test_max_length, return_tensors="pt", pad_to_max_length=True)

      summaries = model_set.model.generate(
       input_ids = sent_ids["input_ids"].to(model_set.device),
       attention_mask = sent_ids["attention_mask"].to(model_set.device),
       num_beams=4,
       length_penalty=2.0,
       max_length=142,  # +2 from original because we start at step=1 and stop before max_length
       min_length=0,  # +1 from original because we start at step=1
       no_repeat_ngram_size=3,
       early_stopping=True,
       do_sample=False,
      )
      
      model_predictions = model_set.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
      for j in range(len(sentences)):
        print('-- evaluated',j,'pair of',batch_idx+1,'batch is following:')
        print('-- source sentence is:')
        print(sentences[j])
        print('-- reference sentence is:')
        print(headlines[j])
        ref = [headlines[j]]
        print('-- generated sentence is:')
        print(model_predictions[j])
        candi = [model_predictions[j]]
        P, R, F1 = score(candi, ref, lang="en", verbose=True)
        print(f"System level F1 score: {F1.mean():.3f}")
        F1_score = F1.numpy()
        nums.append(F1_score)

   np_mean =  np.mean(nums)
   print('-- current bert score(trained):',np_mean)

   return nums,np_mean

## original model BLEU
def original_model_BLEU(test_data_path = str,test_batch_size = int,  test_max_length = int):

   model_set.model.eval()

   test_dataloader = dataloader(test_data_path,test_batch_size)
   nums = []

   for batch_idx,(sentences,headlines) in enumerate(test_dataloader):

      sent_ids = model_set.tokenizer.batch_encode_plus(sentences, max_length=test_max_length, return_tensors="pt", pad_to_max_length=True)

      summaries = model_set.model.generate(
       input_ids = sent_ids["input_ids"].to(model_set.device),
       attention_mask = sent_ids["attention_mask"].to(model_set.device),
       num_beams=4,
       length_penalty=2.0,
       max_length=142,  # +2 from original because we start at step=1 and stop before max_length
       min_length=0,  # +1 from original because we start at step=1
       no_repeat_ngram_size=3,
       early_stopping=True,
       do_sample=False,
      )

      model_predictions = model_set.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
      for j in range(len(sentences)):
        print('-- evaluated',j,'pair of',batch_idx+1,'batch is following:')
        print('-- source sentence is:')
        print(sentences[j])
        print('-- reference sentence is:')
        print(headlines[j])
        reference_head_ids = model_set.tokenizer(headlines[j])
        reference_tokens = reference_head_ids['input_ids']
        str1_reference = []
        for k in range(len(reference_tokens)):
          str1 = str(reference_tokens[k])
          str1_reference.append(str1)
          reference = [str1_reference]

        print('-- generated sentence is:')
        print(model_predictions[j])
        candidate_head_ids = model_set.tokenizer(model_predictions[j])
        candidate_tokens = candidate_head_ids['input_ids']
        str2_candidate = []
        for l in range(len(candidate_tokens)):
          str2 = str(candidate_tokens[l])
          str2_candidate.append(str2)
          candidate = str2_candidate
        score = sentence_bleu(reference, candidate)
        print(score)
        nums.append(score)

   np_mean = np.mean(nums)
   print('-- current BLEU score:',np_mean)
   return nums,np_mean

## original model bert
def original_model_bert(test_data_path = str,test_batch_size = int,  test_max_length = int):

   model_set.model.eval()

   test_dataloader = dataloader(test_data_path,test_batch_size)
   nums = []

   for batch_idx,(sentences,headlines) in enumerate(test_dataloader):

      sent_ids = model_set.tokenizer.batch_encode_plus(sentences, max_length=test_max_length, return_tensors="pt", pad_to_max_length=True)

      summaries = model_set.model.generate(
        input_ids = sent_ids["input_ids"].to(model_set.device),
        attention_mask = sent_ids["attention_mask"].to(model_set.device),
        num_beams=4,
        length_penalty=2.0,
        max_length=142,  # +2 from original because we start at step=1 and stop before max_length
        min_length=0,  # +1 from original because we start at step=1
        no_repeat_ngram_size=3,
        early_stopping=True,
        do_sample=False,
      )

      model_predictions = model_set.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
      for j in range(len(sentences)):
        print('-- evaluated',j,'pair of',batch_idx+1,'batch is following:')
        print('-- source sentence is:')
        print(sentences[j])
        print('-- reference sentence is:')
        print(headlines[j])
        reference = [headlines[j]]
        print('-- generated sentence is:')
        print(model_predictions[j])
        candi = [model_predictions[j]]
        P,R,F1 = score(candi,reference,lang="en",verbose=True)
        print(f"System level F1 score: {F1.mean():.3f}")
        F1_score = F1.numpy()
        nums.append(F1_score)

   np_mean = np.mean(nums)
   print('-- current bert score(original):',np_mean)
   return nums,np_mean
