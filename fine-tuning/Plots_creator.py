import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import random

## Learning- and Validation-Loss Plot
def LV_plot(train_loss = list#, valu_loss = list
            ):
  lc_x = []
  lc_y = []
  #vc_x = []
  #vc_y = []
  fig, ax = plt.subplots()
 # print(len(train_loss))
 # print(len(valu_loss))
  for i in range(len(train_loss)):
   # if i % 500 == 0 : # blank
    if i % 1000 == 0: # normal
   # if i % 338 == 0: # Quora
      lc_y.append(train_loss[i]/5)
  for j in range(len(lc_y)):
    lc_x.append(j)
  #for k in range(len(valu_loss)):
   # if k % 500 == 0 : # blank
    #if k % 1000 == 0: # normal
   # if k % 338 == 0: # Quora
   #   vc_y.append(valu_loss[k]/5)
  #for l in range(len(vc_y)):
  #    vc_x.append(l)
  ax.plot(lc_x,lc_y, 'g', label='training loss')
  #ax.plot(vc_x,vc_y, 'r', label='validation loss')
  ax.set_xlabel('epochs')
  ax.set_ylabel('loss')
  ax.set_title('Learning-Loss')
  ax.legend()

  # normal
  ax.set_xlim(0,30)
  ax.set_ylim(0,0.05)
  my_x_ticks = np.arange(0, 30,5)
  my_y_ticks = np.arange(0, 0.05, 0.01)
  epoch_numbers = ['1','2','3','4','5','6']

  # blank
 # ax.set_xlim(0,30)
 # ax.set_ylim(0,0.05)
 # my_x_ticks = np.arange(0, 30,10)
 # my_y_ticks = np.arange(0, 0.05, 0.01)
 # epoch_numbers = ['1','2','3']

  # Quora
 # ax.set_ylim(0,0.1)
 # ax.set_xlim(0,72)
 # my_x_ticks = np.arange(0, 72,12) # Quora
 # my_y_ticks = np.arange(0, 0.1, 0.02) # Quora
 # epoch_numbers = ['1','2','3','4','5','6']

  plt.xticks(my_x_ticks,epoch_numbers)
  plt.yticks(my_y_ticks)
  plt.show()
  plt.savefig('/home/yiwang/Datasets/model/LV_plot.png')
