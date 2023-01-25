import model_set
from model_set import Sent_Comp_Dataset, collate_batch, Valu_Sent_Comp_Dataset, valu_collate_batch
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

def model_train(train_data_path = str, epoch = int, train_batch_size = int, model_save_path = str, train_max_length = int
                , learning_rate = float):

  #model statu
  model_set.model.train(mode=True) # By operation of dropout and batch normalization, to avoid overfitting

  # optimizer
  # optimizer=optim.SGD(model_device_set.model.parameters(),lr=learning_rate,momentum=0.9) 
  optimizer = torch.optim.AdamW(model_set.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

  # loss function
  train_loss = []
  valuation_loss = []

  for epoch in range(epoch):
    random_idx = random.randint(0,9)
    running_loss = 0.0
    scd_train = Sent_Comp_Dataset(train_data_path,random_idx)
    train_dataloader = DataLoader(scd_train, batch_size=train_batch_size, shuffle=True, collate_fn=collate_batch, drop_last=False)
    print('-- Starting Train')
    scd_valu = Valu_Sent_Comp_Dataset(train_data_path,random_idx)
    valu_dataloader = DataLoader(scd_train, batch_size=train_batch_size, shuffle=True, collate_fn=valu_collate_batch, drop_last=False)

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
        loss = model_set.model(**sent_ids, labels = y_ids).loss
        loss.backward()
        train_loss.append(loss.item())

        # optimizer
        optimizer.step()

        running_loss = 0.0
        running_loss += loss.item()
        # print every 5 mini-batches
        if batch_idx % 5 == 4:
            print('[%d,%5d] loss: %.3f' %(epoch +1, batch_idx + 1, running_loss /5 ))
            running_loss = 0.0

        #valuation 
        model_set.model.eval() # By operation of dropout and batch normalization, to avoid overfitting

        random_batch_idx = random.randint(0,620)
        for random_batch_idx,(sentences,headlines) in enumerate(valu_dataloader):
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
          break

    print('-- Finish epoch', '%d' % (epoch + 1))

    # model saving
    torch.save(model_set.model.state_dict(), model_save_path) # saves only the model parameters
    print('-- Finish Training')
  return train_loss, valuation_loss
