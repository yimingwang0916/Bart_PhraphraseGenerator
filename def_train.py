import model_device_set
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel, AutoModel

def model_train(train_data_path = str, epoch = int, train_batch_size = int, model_save_path = str, train_max_length = int
                , learning_rate = float,random_idx = int ):

  #model statu
  model_device_set.model.train(mode=True) # By operation of dropout and batch normalization, to avoid overfitting

  # scd_train = Sent_Comp_Dataset(train_data_path,random_idx)
  # train_dataloader = DataLoader(scd_train, batch_size=train_batch_size, shuffle=True, collate_fn=collate_batch, drop_last=False) 

  # optimizer
  # optimizer=optim.SGD(model_device_set.model.parameters(),lr=learning_rate,momentum=0.9) 
  optimizer = torch.optim.AdamW(model_device_set.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

  # loss function
  # loss = nn.CrossEntropyLoss()
  train_loss = []
  valuation_loss = []

  for epoch in range(epoch):

    running_loss = 0.0
    scd_train = Sent_Comp_Dataset(train_data_path,random_idx)
    train_dataloader = DataLoader(scd_train, batch_size=train_batch_size, shuffle=True, collate_fn=collate_batch, drop_last=False)
    print('-- Starting Train')
    for batch_idx,(sentences,headlines) in enumerate(train_dataloader):
        sent_ids = model_device_set.tokenizer.batch_encode_plus(sentences, max_length=train_max_length, return_tensors="pt", pad_to_max_length=True).to(model_device_set.device)
        head_ids = model_device_set.tokenizer.batch_encode_plus(headlines, max_length=train_max_length, return_tensors="pt", pad_to_max_length=True).to(model_device_set.device)


        # further process
        pad_token_id = model_device_set.tokenizer.pad_token_id
        y = head_ids['input_ids']
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100

        # forward + backward
        optimizer.zero_grad()

        output = model_device_set.model(
          input_ids =  sent_ids['input_ids'].to(model_device_set.device),
          attention_mask = sent_ids['attention_mask'].to(model_device_set.device),
          decoder_input_ids = y_ids.to(model_device_set.device),
         # decoder_input_ids = y.to(model_device_set.device),
         # labels = lm_labels.to(model_device_set.device),
            )

        optimizer.zero_grad()

        # loss function
        loss = model_device_set.model(**sent_ids, labels = y_ids).loss
        loss.backward()
        train_loss.append(loss.item())

        # CrossEntropy
        # logits = output.logits
        # vocab_size_logits = model_device_set.tokenizer.vocab_size
        # logits_view = logits.view(train_batch_size, vocab_size_logits, -1)
        # loss(logits_view, y_ids).backward()

        # optimizer
        optimizer.step()
        running_loss = 0.0
        running_loss += loss.item()
        # running_loss += loss(logits_view, y_ids).item()
        # print every 5 mini-batches
        if batch_idx % 5 == 4:
          print('-- [%d,%5d] loss: %.3f' %(epoch +1, batch_idx + 1, running_loss /5 ))
          running_loss = 0.0

    print('-- Finish epoch', '%d' % (epoch + 1))

    # model saving
    torch.save(model_device_set.model.state_dict(), model_save_path) # saves only the model parameters
    print('-- Finish Training')

    #valuation 
    print('-- Strating Valuation')
    model_device_set.model.eval() # By operation of dropout and batch normalization, to avoid overfitting

    scd_valu = Valu_Sent_Comp_Dataset(train_data_path,random_idx)
    valu_dataloader = DataLoader(scd_train, batch_size=train_batch_size, shuffle=True, collate_fn=valu_collate_batch, drop_last=False)

    for batch_idx,(sentences,headlines) in enumerate(train_dataloader):
      sent_ids = model_device_set.tokenizer.batch_encode_plus(sentences, max_length=train_max_length, return_tensors="pt", pad_to_max_length=True).to(model_device_set.device)
      head_ids = model_device_set.tokenizer.batch_encode_plus(headlines, max_length=train_max_length, return_tensors="pt", pad_to_max_length=True).to(model_device_set.device)

      # further process
      pad_token_id = model_device_set.tokenizer.pad_token_id
      y = head_ids['input_ids']
      y_ids = y[:, :-1].contiguous()
      lm_labels = y[:, 1:].clone()
      lm_labels[y[:, 1:] == pad_token_id] = -100

      # loss functions
      valu_loss = model_device_set.model(**sent_ids, labels = y_ids).loss
      valuation_loss.append(valu_loss.item())
    print('-- Finish Valuation')
  return train_loss, valuation_loss
