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

def train(train_data_path = str, epoch = int, train_batch_size = int, model_save_path = str, train_max_length = int):

  # model statu
  model.train(mode=True) # By operation of dropout and batch normalization, to avoid overfitting
 
  datalist = []
  for n in os.listdir(train_data_path):
    if os.path.splitext(n)[1] == '.json':   
        datalist.append(n)

  os.chdir(train_data_path) 
  for i in range(len(datalist)):

    scd_train = Sent_Comp_Dataset(datalist[i])
    train_dataloader = DataLoader(scd_train, batch_size=train_batch_size, shuffle=True, collate_fn=collate_batch, drop_last=False) 
  
    for epoch in range(epoch):
      running_loss = []

      for batch_idx,(sentences,headlines) in enumerate(train_dataloader):
        sent_ids = tokenizer.batch_encode_plus(sentences, max_length=train_max_length, return_tensors="pt", pad_to_max_length=True)
        head_ids = tokenizer.batch_encode_plus(headlines, max_length=train_max_length, return_tensors="pt", pad_to_max_length=True)


        # further process
        pad_token_id = tokenizer.pad_token_id
        y = head_ids['input_ids']
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()   
        lm_labels[y[:, 1:] == pad_token_id] = -100

        # forward + backward + optimize
        output = model(
          input_ids = sent_ids['input_ids'].to(device),
          attention_mask = sent_ids['attention_mask'].to(device),
          decoder_input_ids = y_ids.to(device),
         # lm_labels = lm_labels.to(device),
            )
        
        # loss function
        loss = model(**sent_ids, labels = y_ids).loss
        #criterion = nn.CrossEntropyLoss()
        #loss = criterion(predicted_class_id, y_ids)
        loss.backward()

        # optimizer
        optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)  
        optimizer.step()  

        running_loss = 0.0
        running_loss += loss.item()
        # print every 20 mini-batches
        if batch_idx % 20 ==19:
          print('[%d,%5d] loss: %.3f' %(epoch +1, batch_idx + 1, running_loss /20 ))
          running_loss = 0.0

      print('Finish epoch', '%d' % (epoch + 1))

  # model saving
  torch.save(model.state_dict(), model_save_path) # saves only the model parameters
  print('Finished Training')
  
 

def test(test_data_path=str, test_batch_size = int, model_save_path = str, test_max_length = int
         ):
   
  model.load_state_dict(torch.load(model_save_path)) # loads only the model parameters
  model.eval()

  scd_test = Sent_Comp_Dataset(test_data_path)
  test_dataloader = DataLoader(scd_test, batch_size=test_batch_size, shuffle=True, collate_fn=collate_batch, drop_last=False)

  for batch_idx,(sentences,headlines) in enumerate(test_dataloader):  
    
    sent_ids = tokenizer.batch_encode_plus(sentences, max_length=test_max_length
                                           , return_tensors="pt", pad_to_max_length=True) 

    summaries = model.generate(
      input_ids=sent_ids["input_ids"].to(device),
      attention_mask=sent_ids["attention_mask"].to(device),
      num_beams=4,
      length_penalty=2.0,
      max_length=142,  # +2 from original because we start at step=1 and stop before max_length
      min_length=56,  # +1 from original because we start at step=1
      no_repeat_ngram_size=3,
      early_stopping=True,
      do_sample=False,
    )  # change these arguments if you want

  dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
  print(dec)
