import matplotlib
import pylab
import matplotlib.pyplot as plt
import numpy as np

def learning_curve(train_loss = list):
        x = []
        y = []
        for i in range(len(train_loss)):
          if i % 600 == 0:
            y.append(train_loss[i])
          else:
            break
        for j in range(len(y)):
            x.append(j+1)
        plt.plot(x, y, color = 'r',marker = 'o',linestyle = 'dashed')
        plt.axis([0, len(y), 0, 0.1])
        plt.xlabel('numbers of batches')
        plt.ylabel('loss')
        plt.title('learning-curve')
        plt.show()
        plt.savefig('/home/yiwang/Datasets/model/learning_curve.png')

def valuation_curve(valu_loss = list):
        x = []
        y = []
        for i in range(len(valu_loss)):
          if i % 30 == 0:
            y.append(valu_loss[i])
          else:
            break
        for j in range(len(y)):
            x.append(j+1)
        plt.plot(x, y, color = 'r',marker = 'o',linestyle = 'dashed')
        plt.axis([0, len(y), 0, 0.05])
        plt.xlabel('numbers of batches')
        plt.ylabel('loss')
        plt.title('valuation-curve')
        plt.show()
        plt.savefig('/home/yiwang/Datasets/model/valuation_curve.png')
