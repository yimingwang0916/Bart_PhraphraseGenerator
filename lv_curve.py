import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np

def curve(train_loss = list, valu_loss = list):
  lc_x = []
  lc_y = []
  vc_x = []
  vc_y = []
  #lc_y
  for i in range(len(train_loss)):
    if i % 60 == 0:
      lc_y.append(train_loss[i]/5)
  #lc_x
  for j in range(len(lc_y)):
    lc_x.append(j)
  #vc_y
  for k in range(len(valu_loss)):
    if k % 60 == 0:
      vc_y.append(valu_loss[k]/5)
  #vc_x
  for l in range(len(vc_y)):
      vc_x.append(l)

  # plot the data
  fig, ax = plt.subplots() 

  ax.plot(lc_x,lc_y, label='learning_curve')
  ax.plot(vc_x,vc_y, label='quadratic') 
  ax.set_xlabel('batches') 
  ax.set_ylabel('loss') 
  ax.set_title('learning- and valuation-curve') 
  ax.legend() 
  ax.set_xlim([0, 100])
  ax.set_ylim([0, 0.1])
  # display the plot
  plt.show()
  plt.savefig('/home/yiwang/Datasets/model/curve.png')
