import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np

## Learning- and Valiation-Loss Plot
def LV_plot(train_loss = list, valu_loss = list):
  lc_x = []
  lc_y = []
  vc_x = []
  vc_y = []
  #lc_y
  for i in range(len(train_loss)):
    if i % 1125 == 0:
      lc_y.append(train_loss[i]/5)
  #lc_x
  for j in range(len(lc_y)):
    lc_x.append(j)
  #vc_y
  for k in range(len(valu_loss)):
    if k % 1125 == 0:
      vc_y.append(valu_loss[k]/5)
  #vc_x
  for l in range(len(vc_y)):
      vc_x.append(l)
  # plot the data
  fig, ax = plt.subplots()
  ax.plot(lc_x,lc_y, 'g', label='learning_curve')
  ax.plot(vc_x,vc_y, 'r', label='valuation_curve')
  ax.set_xlabel('epochs')
  ax.set_ylabel('loss')
  ax.set_title('learning- and valuation-curve')
  ax.legend()
  ax.set_xlim(0,40)
  ax.set_ylim(0,0.1)
  my_x_ticks = np.arange(0, 40, 5)
  my_y_ticks = np.arange(0, 0.1, 0.02)
  epoch_numbers = ['1','2','3','4','5','6','7','8']
  plt.xticks(my_x_ticks,epoch_numbers)
  plt.yticks(my_y_ticks)
  # display the plot
  plt.show()
  plt.savefig('/home/yiwang/Datasets/model/LV_plot.png')

## Ratio Plot
def RL_plot(ratio_OS = list, ratio_RS = list):

    OS_x = []
    OS_y = []
    RS_x = []
    RS_y = []

    for q in range(len(ratio_OS)):
       if q % 2000 ==0:
    #  if q % 2000 ==0:
            os = np.mean(ratio_OS[(q-2000):q])
           # os = np.mean(ratio_OS[(q-2000):q])
            OS_y.append(os)
    for r in range(len(OS_y)):
        OS_x.append(r)

    for i in range(len(ratio_RS)):
        if i % 2000 ==0:
     #  if i % 2000 ==0:
            rs = np.mean(ratio_RS[(i-2000):i])
           # rs = np.mean(ratio_RS[(i-2000):i])
            RS_y.append(rs)
    for j in range(len(RS_y)):
        RS_x.append(j)

    # plot the data
    fig, ax = plt.subplots()
    ax.plot(OS_x,OS_y, 'y--', label= 'outout/source')
    ax.plot(RS_x,RS_y, 'r--', label= 'ref/source')
    ax.set_xlabel('epochs')
    ax.set_ylabel('ratio')
    ax.set_title('Lengths Ratio')
    ax.legend()
   # ax.set_xlim(0,50)
    ax.set_xlim(0,90)
    ax.set_ylim(0,0.5)
   # my_x_ticks = np.arange(0,50,5)
    my_x_ticks = np.arange(0,90,10)
    my_y_ticks = np.arange(0, 0.5, 0.1)
   # epoch_numbers = ['1','2','3','4','5','6','7','8']
    plt.xticks(my_x_ticks
   #        ,epoch_numbers
           )
    plt.yticks(my_y_ticks)
    # display the plot
    plt.show()
    plt.savefig('/home/yiwang/Datasets/model/RL_plot.png')

## Learning-Loss and Lengths Plot
def LL_plot(source_len = list, output_len = list, reference_len = list):
  sl_x = []
  sl_y = []
  ol_x = []
  ol_y = []
  rl_x = []
  rl_y = []

  #sl
  for k in range(len(source_len)):
    if k % 2000 == 0:
      sl = np.mean(source_len[(k-2000):k])
      sl_y.append(sl)
  for l in range(len(sl_y)):
      sl_x.append(l)

  #ol
  for m in range(len(output_len)):
    if m % 2000 == 0:
      ol = np.mean(output_len[(m-2000):m])
      ol_y.append(ol)
  for n in range(len(ol_y)):
      ol_x.append(n)

  #rl
  for o in range(len(reference_len)):
    if o % 2000 == 0:
        rl = np.mean(reference_len[(o-2000):o])
        rl_y.append(rl)
  for p in range(len(rl_y)):
      rl_x.append(p)

  # plot the data
  fig, ax = plt.subplots()

 # ax.plot(lc_x,lc_y, 'k--', label='learning_curve')
  ax.plot(sl_x,sl_y, 'r--', label='source_length')
  ax.plot(ol_x,ol_y, 'g--', label='output_length')
  ax.plot(rl_x,rl_y, 'm--', label='reference_length')
  ax.set_xlabel('epochs')
  ax.set_ylabel('words')
  ax.set_title('Lengths-curve')
  ax.legend()
  #ax.set_xlim(0,50)
  ax.set_xlim(0,90)
  ax.set_ylim(30,60)
  #my_x_ticks = np.arange(0, 50, 5)
  my_x_ticks = np.arange(0,90,10)
  my_y_ticks = np.arange(30, 60, 10)
 # epoch_numbers = ['1','2','3','4','5','6','7','8']
  plt.xticks(my_x_ticks
 #         ,epoch_numbers
          )
  plt.yticks(my_y_ticks)
  # display the plot
  plt.show()
  plt.savefig('/home/yiwang/Datasets/model/LL_plot.png')



## Results' from original and trained model
def OT_plot(original_model_result = list, trained_model_result = list):

    # Probability distribution
    trained_model_result_0 = []
    trained_model_result_005 = []
    trained_model_result_01 = []
    trained_model_result_015 = []
    trained_model_result_02 = []
    trained_model_result_025 = []
    trained_model_result_03 = []
    trained_model_result_035 = []
    trained_model_result_04 = []
    trained_model_result_045 = []
    trained_model_result_05 = []
    trained_model_result_055 = []
    trained_model_result_06 = []
    trained_model_result_065 = []
    trained_model_result_07 = []
    trained_model_result_075 = []
    trained_model_result_08 = []
    trained_model_result_085 = []
    trained_model_result_09 = []
    trained_model_result_095 = []
    trained_model_result_1 = []
    trained_model_pro = []

    for i, val in enumerate(trained_model_result):
        if 0 <= val and val <0.05:
            trained_model_result_0.append(1)
        if 0.05 <= val and val <0.1:
            trained_model_result_005.append(1)
        if 0.1 <= val and val <0.15:
            trained_model_result_01.append(1)
        if 0.15 <= val and val <0.2:
            trained_model_result_015.append(1)
        if 0.2 <= val and val <0.25:
            trained_model_result_02.append(1)
        if 0.25 <= val and val <0.3:
            trained_model_result_025.append(1)
        if 0.3 <= val and val <0.35:
            trained_model_result_03.append(1)
        if 0.35 <= val and val <0.4:
            trained_model_result_035.append(1)
        if 0.4 <= val and val <0.45:
            trained_model_result_04.append(1)
        if 0.45 <= val and val <0.5:
            trained_model_result_045.append(1)
        if 0.5 <= val and val <0.55:
            trained_model_result_05.append(1)
        if 0.55 <= val and val <0.6:
            trained_model_result_055.append(1)
        if 0.6 <= val and val <0.65:
            trained_model_result_06.append(1)
        if 0.65 <= val and val <0.7:
            trained_model_result_065.append(1)
        if 0.7 <= val and val <0.75:
            trained_model_result_07.append(1)
        if 0.75 <= val and val <0.8:
            trained_model_result_075.append(1)
        if 0.8 <= val and val <0.85:
            trained_model_result_08.append(1)
        if 0.85 <= val and val <0.9:
            trained_model_result_085.append(1)
        if 0.9 <= val and val <0.95:
            trained_model_result_09.append(1)
        if 0.95 <= val and val <1:
            trained_model_result_095.append(1)
        if 1 == val:
            trained_model_result_1.append(1)

    trained_model_pro = [len(trained_model_result_0),
        len(trained_model_result_005),
        len(trained_model_result_01),
        len(trained_model_result_015),
        len(trained_model_result_02),
        len(trained_model_result_025),
        len(trained_model_result_03),
        len(trained_model_result_035),
        len(trained_model_result_04),
        len(trained_model_result_045),
        len(trained_model_result_05),
        len(trained_model_result_055),
        len(trained_model_result_06),
        len(trained_model_result_065),
        len(trained_model_result_07),
        len(trained_model_result_075),
        len(trained_model_result_08),
        len(trained_model_result_085),
        len(trained_model_result_09),
        len(trained_model_result_095),
        len(trained_model_result_1)]

    # Probability distribution
    original_model_result_0 = []
    original_model_result_005 = []
    original_model_result_01 = []
    original_model_result_015 = []
    original_model_result_02 = []
    original_model_result_025 = []
    original_model_result_03 = []
    original_model_result_035 = []
    original_model_result_04 = []
    original_model_result_045 = []
    original_model_result_05 = []
    original_model_result_055 = []
    original_model_result_06 = []
    original_model_result_065 = []
    original_model_result_07 = []
    original_model_result_075 = []
    original_model_result_08 = []
    original_model_result_085 = []
    original_model_result_09 = []
    original_model_result_095 = []
    original_model_result_1 = []
    original_model_pro = []

    for j, val in enumerate(original_model_result):
        if 0 <= val and val <0.05:
            original_model_result_0.append(1)
        if 0.05 <= val and val <0.1:
            original_model_result_005.append(1)
        if 0.1 <= val and val <0.15:
            original_model_result_01.append(1)
        if 0.15 <= val and val <0.2:
            original_model_result_015.append(1)
        if 0.2 <= val and val <0.25:
            original_model_result_02.append(1)
        if 0.25 <= val and val <0.3:
            original_model_result_025.append(1)
        if 0.3 <= val and val <0.35:
            original_model_result_03.append(1)
        if 0.35 <= val and val <0.4:
            original_model_result_035.append(1)
        if 0.4 <= val and val <0.45:
            original_model_result_04.append(1)
        if 0.45 <= val and val <0.5:
            original_model_result_045.append(1)
        if 0.5 <= val and val <0.55:
            original_model_result_05.append(1)
        if 0.55 <= val and val <0.6:
            original_model_result_055.append(1)
        if 0.6 <= val and val <0.65:
            original_model_result_06.append(1)
        if 0.65 <= val and val <0.7:
            original_model_result_065.append(1)
        if 0.7 <= val and val <0.75:
            original_model_result_07.append(1)
        if 0.75 <= val and val <0.8:
            original_model_result_075.append(1)
        if 0.8 <= val and val <0.85:
            original_model_result_08.append(1)
        if 0.85 <= val and val <0.9:
            original_model_result_085.append(1)
        if 0.9 <= val and val <0.95:
            original_model_result_09.append(1)
        if 0.95 <= val and val <1:
            original_model_result_095.append(1)
        if 1 == val:
            original_model_result_1.append(1)

    original_model_pro = [
            len(original_model_result_0),
            len(original_model_result_005),
            len(original_model_result_01),
            len(original_model_result_015),
            len(original_model_result_02),
            len(original_model_result_025),
            len(original_model_result_03),
            len(original_model_result_035),
            len(original_model_result_04),
            len(original_model_result_045),
            len(original_model_result_05),
            len(original_model_result_055),
            len(original_model_result_06),
            len(original_model_result_065),
            len(original_model_result_07),
            len(original_model_result_075),
            len(original_model_result_08),
            len(original_model_result_085),
            len(original_model_result_09),
            len(original_model_result_095),
            len(original_model_result_1)]

    om_x = [0.025,0.075,0.125,0.175,0.225,0.275,0.325,0.375,0.425,0.475,0.525,0.575,0.625,0.675,0.725,0.775,0.825,0.875,0.925,0.975,1]
    om_y = original_model_pro
    tm_x = [0.025,0.075,0.125,0.175,0.225,0.275,0.325,0.375,0.425,0.475,0.525,0.575,0.625,0.675,0.725,0.775,0.825,0.875,0.925,0.975,1]
    tm_y = trained_model_pro

    # plot the data
    fig, ax = plt.subplots()

    ax.set_xlabel('Score')
    ax.set_ylabel('Quantity')
    ax.set_title('Result-Score distribution')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 3500])

    my_x_ticks = np.arange(0, 1, 0.1)
    my_y_ticks = np.arange(0, 3500, 500)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    ax.plot(om_x,om_y,'b--',label='pre-trained model')
    ax.plot(tm_x,tm_y,'r--',label='after training')
   # plt.plot(om_x,om_y,'bo-',tm_x,tm_y,'r^-')
    ax.legend()

    # display the plot
    plt.show()
    plt.savefig('/home/yiwang/Datasets/model/OT_plot.png')


def result_means(original_model_means = int, trained_model_means = int):
  # plot the data
  fig, ax = plt.subplots()
  ax.set_xlabel('')
  ax.set_ylabel('Score')
  ax.set_title('result-score-means')
  ax.set_xlim([0, 1])
  ax.set_ylim([0, 1])
  ax.axhline(y=original_model_means, color='b', linestyle='--',label='pre-trained model')
  ax.axhline(y=trained_model_means, color='r', linestyle='--',label='after training')
  ax.legend()
  # display the plot
  plt.show()
  plt.savefig('/home/yiwang/Datasets/model/result_means.png')
