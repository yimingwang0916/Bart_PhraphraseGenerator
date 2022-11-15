import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np

def curve(original_model_result = list, trained_model_result = list):
  om_x = []
  om_y = []
  tm_x = []
  tm_y = []
  #om_y
  for i in range(len(original_model_result)):
    if i % 3 == 0:
      om_y.append(original_model_result[i])
  #om_x
  for j in range(len(om_y)):
    om_x.append(j)
  #vc_y
  for k in range(len(trained_model_result)):
    if k % 3 == 0:
      tm_y.append(trained_model_result[k])
  #vc_x
  for l in range(len(tm_y)):
      tm_x.append(l)

  # plot the data
  fig, ax = plt.subplots() 

  ax.plot(om_x,om_y, label='pre-trained model')
  ax.plot(tm_x,tm_y, label='after training') 
  ax.set_xlabel('batches') 
  ax.set_ylabel('Bleu score') 
  ax.set_title('result') 
  ax.legend() 
  ax.set_xlim([0, 100])
  ax.set_ylim([0, 1])
  # display the plot
  plt.show()
  plt.savefig('/home/yiwang/Datasets/model/result_curve.png')
