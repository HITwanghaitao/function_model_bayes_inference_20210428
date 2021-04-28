# 绘制后验分布曲线
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import os

params = ['Lgrad','vc','h','m','rou']
main_path = os.getcwd()
time_file_name = 'Time2021_04_28_08_15_12_function_model_bayes_inference'
path = main_path + '\\' + time_file_name

for k in range(5):
    theta = io.loadmat(path + '\\' + params[k] +'_posterior.mat')
    plt.figure(k)
    for j in range(4):
        x = theta[params[k]+'x'][j, :]
        y = theta[params[k]+'y'][j,:]
        plt.plot(x, y, label=str(j+1))

    plt.yticks([])
    plt.legend()
    plt.title('posteriori distribution of ' + params[k] + ' for 4 times')
    plt.savefig(path + '\\' + params[k] + '.png')

plt.show()

