import model_set
from model_set import Sent_Comp_Dataset, collate_batch, Quora_Sent_Comp_Dataset, Quora_collate_batch
import numpy as np
from numpy import mean
import random
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

def model_train(train_data_path = str, valu_data_path = str,  epoch = int, train_batch_size = int, model_save_path = str, train_max_length = int
                , learning_rate = float):

  #model statu
  model_set.model.train(mode=True) # By operation of dropout and batch normalization, to avoid overfitting

  # optimizer
  # optimizer=optim.SGD(model_device_set.model.parameters(),lr=learning_rate,momentum=0.9) 
  optimizer = torch.optim.AdamW(model_set.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

  # loss function
  train_loss = []
  valuation_loss = []
  train_loss = []
  source_len = [] 
  output_len = []
  reference_len = []
  ratio_OS = []
  ratio_RS = []
  
  for epoch in range(epoch):
    running_loss = 0.0
    scd_train = Sent_Comp_Dataset(train_data_path)
    train_dataloader = DataLoader(scd_train, batch_size=train_batch_size, shuffle=True, collate_fn=collate_batch, drop_last=False)
    print('-- Starting Train')
    scd_valu = Sent_Comp_Dataset(valu_data_path)
    valu_dataloader = DataLoader(scd_valu, batch_size=train_batch_size, shuffle=True, collate_fn=collate_batch, drop_last=False)

    for batch_idx,(sentences,headlines) in enumerate(train_dataloader):
        sent_ids = model_set.tokenizer.batch_encode_plus(sentences, max_length=train_max_length, return_tensors="pt", pad_to_max_length=True).to(model_set.device)
        head_ids = model_set.tokenizer.batch_encode_plus(headlines, max_length=train_max_length, return_tensors="pt", pad_to_max_length=True).to(model_set.device)


        # further process
        pad_token_id = model_set.tokenizer.pad_token_id
        y = head_ids['input_ids']
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100

        # forward + backward
        optimizer.zero_grad()

        output = model_set.model(
          input_ids =  sent_ids['input_ids'].to(model_set.device),
          attention_mask = sent_ids['attention_mask'].to(model_set.device),
          decoder_input_ids = y_ids.to(model_set.device),
         # decoder_input_ids = y.to(model_set.device),
         # labels = lm_labels.to(model_et.device),
            )

        optimizer.zero_grad()

        # loss function
        print('121212')
        loss = model_set.model(**sent_ids, labels = y_ids).loss
        print(loss) # tensor(15.7336, device='cuda:3', grad_fn=<NllLossBackward0>)
        print(loss.requires_grad)
        print(type(loss))
        print(loss.shape)
        print(loss.size())
        print('232323')
        loss.backward()
        print('343434')
        train_loss.append(loss.item())

        # optimizer
        optimizer.step()

        running_loss = 0.0
        running_loss += loss.item()
        # print every 5 mini-batches
        if batch_idx % 5 == 4:
            print('[%d,%5d] loss: %.5f' %(epoch +1, batch_idx + 1, running_loss /5 ))
            running_loss = 0.0
        
        #valuation 
        model_set.model.eval() # By operation of dropout and batch normalization, to avoid overfitting
        random_idx = random.randint(0,150)
        for idx,(sentences,headlines) in enumerate(valu_dataloader):
            
            if idx == random_idx:
                  sent_ids = model_set.tokenizer.batch_encode_plus(sentences, max_length=train_max_length, return_tensors="pt", pad_to_max_length=True).to(model_set.device)
                  head_ids = model_set.tokenizer.batch_encode_plus(headlines, max_length=train_max_length, return_tensors="pt", pad_to_max_length=True).to(model_set.device)

                  # further process
                  pad_token_id = model_set.tokenizer.pad_token_id
                  y = head_ids['input_ids']
                  y_ids = y[:, :-1].contiguous()
                  lm_labels = y[:, 1:].clone()
                  lm_labels[y[:, 1:] == pad_token_id] = -100

                  # loss functions
                  valu_loss = model_set.model(**sent_ids, labels = y_ids).loss
                  valuation_loss.append(valu_loss.item())
                  
                  summaries = model_set.model.generate(
                          input_ids = sent_ids["input_ids"].to(model_set.device),
                          attention_mask = sent_ids["attention_mask"].to(model_set.device),
                          num_beams=4,                                    
                          length_penalty=2.0,max_length=142,  # +2 from original because we start at step=1 and stop before max_length                                                          
                          min_length=0,  # +1 from original because we start at step=1                                                                        
                          no_repeat_ngram_size=3,
                          early_stopping=True,
                          do_sample=False,
                          )

                  # length info 
                  model_predictions = model_set.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                 # print(len(sentences))
                 # print(len(headlines))
                 # print(len(model_predictions))
                  for h in range(len(sentences)):
                      sentences_len = sentences[h].split()
                      source_len.append(len(sentences_len))
                      model_predictions_len = model_predictions[h].split()
                      output_len.append(len(model_predictions_len))
                      headlines_len = headlines[h].split()
                      reference_len.append(len(headlines_len))                                                                      
                      ratio_os = len(model_predictions_len)/len(sentences_len)
                      ratio_OS.append(ratio_os)
                      ratio_rs = len(headlines_len)/len(sentences_len)
                      ratio_RS.append(ratio_rs)
                  
    print('-- Finish epoch', '%d' % (epoch + 1))
    # model saving
    torch.save(model_set.model.state_dict(), model_save_path) # saves only the model parameters
    print('-- Finish Training')
  return train_loss, valuation_loss, source_len,output_len,reference_len,ratio_OS, ratio_RS
