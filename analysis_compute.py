# 全流程第二步 2.
'''
5*5
0.403000000000000	0	0.200000000000000	0.600000000000000	0
0.400279810126558	0	0.363000000000000	0.442000000000000	0
0.404263505260806	0	0.383000000000000	0.424000000000000	0
0.403719868010295	0	0.387000000000000	0.419000000000000	0
0.403759335961369	0	0.390000000000000	0.417000000000000	0
生成
5*5
0.403000000000000	0	0.200000000000000	0.600000000000000	0
0.400279810126558	0.674985080258663	0.363000000000000	0.442000000000000	19.7500000000000
0.404263505260806	0.313524878611962	0.383000000000000	0.424000000000000	10.2500000000000
0.403719868010295	0.178627297839840	0.387000000000000	0.419000000000000	7.99999999999999
0.403759335961369	0.188420834086659	0.390000000000000	0.417000000000000	6.74999999999999
'''
# 用于将_bayes_analysis.mat 生成 _bayes_analysis_compute.mat_c.mat
import numpy as np
import scipy.io as io
import os

params = ['Lgrad','vc','h','m','rou']
main_path = os.getcwd()
time_file_name = 'Time2021_04_28_08_15_12_function_model_bayes_inference'
path = main_path + '\\' + time_file_name

for k in range(5):
    theta_param = io.loadmat(path + '\\' + params[k] + '_bayes_analysis.mat')
    theta = theta_param[params[k]]
    #print(theta)
    matpath1 = path + '\\' + params[k] + '_bayes_analysis_compute.mat'
    for j in range(4):
        theta[j+1,1] = np.abs(theta[j+1,0] - theta[0,0])/theta[0, 0]*100

        theta[j + 1, 4] = (theta[j+1,3] - theta[j+1,2])/(theta[0,3] - theta[0,2])*100
    io.savemat(matpath1,{params[k]+'_c':theta})
