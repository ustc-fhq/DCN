# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 23:08:14 2019

@author: 孔嘉伟
"""

import matplotlib.pyplot as plt
import os
import socket
import pandas as pd
#import xlrd
import numpy as np
from keras.models import load_model
import json
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import CuDNNLSTM
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
start = time.time()

def creat_dataset(data, look_back):
        dataX, dataY=[], []
        for i in range(len(data)-look_back-predictSampleNum - 1):
            dataX.append(data[i:(i+look_back)])
            dataY.append(data[(i+look_back):(i+look_back + predictSampleNum)])
        return np.array(dataX), np.array(dataY)

def creat_dataset1(data, look_back):
    dataX, dataY=[], []
    for i in range(1):
        dataX.append(data[i:(i+look_back)])
        dataY.append(data[(i+look_back):(i+look_back + predictSampleNum)])
    return np.array(dataX), np.array(dataY)


def csvtolist(pattern):    #CSV转化
    data = pd.read_csv( pattern + '.csv' , usecols=[1])   #traffic1
    data = data.values.tolist()
    return data

#预处理
def preprocess(dataset) :
    dataset = np.array(dataset)
    for i in range(2000//24):
        dataset[i*24+12]=sum(dataset[i*24:i*24+24])
        dataset[i*24:i*24+12] = 0
        dataset[i*24+13:i*24+24] = 0
    return list(dataset)
    # dataset = np.array(dataset)
    # dataset = list(dataset)
    # state = 0     #preprocess
    # for i in range(len(dataset)):
    #     if(state == 0 and dataset[i] > 0) :
    #         start = i
    #         state = 1
    #     if(state == 1 and dataset[i] == 0):
    #         end = i
    #         state = 0
    #         dataset[start] = sum(dataset[start:end])
    #         dataset = np.array(dataset)
    #         dataset[start+1:end] = 0
    #         dataset = list(dataset)
    #     if(state == 1 and dataset[i] > 0 and i == len(dataset) - 1):
    #         end = i + 1
    #         state = 0
    #         dataset[start] = sum(dataset[start:end])
    #         dataset = np.array(dataset)
    #         dataset[start+1:end] = 0
    #         dataset = list(dataset)
    # return dataset
def preprocess1(dataset) :
    dataset = np.array(dataset)
    for i in range(1):
        dataset[i*24+12]=sum(dataset[i*24:i*24+24])
        dataset[i*24:i*24+12] = 0
        dataset[i*24+13:i*24+24] = 0
    return dataset
#预测
def predict(dataset_traffic1_pre,i):
    dataset_traffic1_pre = np.array(dataset_traffic1_pre)
    test_inputdata = np.reshape(dataset_traffic1_pre, (dataset_traffic1_pre.shape[0], 1, dataset_traffic1_pre.shape[1]))
    for k in range(test_inputdata.shape[0]):
        test_temp = np.array([test_inputdata[k, :, :]])
        predict_traffic1 = model[i].predict(test_temp)
    predict_traffic1 = predict_traffic1[0]
    return predict_traffic1

#起始时间
def start_time(predict_node):
    start = 0
    for i in range(len(predict_node)):
        if(predict_node[i] > 0):
            start = i
            break
    return start
           
            
        
    
#训练模型
traffictrainingSampleNum = 2300
cputrainingSampleNum = 2000
look_back = 1200
predictSampleNum = 24
dataset_node_set=[]
start_set=[]
sum_node_set = []

#服务器端
sock_server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock_server.bind(('127.0.0.1',10001))
sock_server.listen(5)

#模型载入
model_node1 = load_model('node1' +'2L' + '.h5')
#model_node2 = load_model('node2' +'2L' + '.h5')
model_node3 = load_model('node3' +'2L' + '.h5')
#model_node4 = load_model('node4' +'2L' + '.h5')
model_node5 = load_model('node5' +'2L' + '.h5')
#model_node6 = load_model('node6' +'2L' + '.h5')
print(time.time())
#model_traffic1 = load_model('traffic1' +'2L' + '.h5')
#model_traffic2 = load_model('traffic2' +'2L' + '.h5')
#model_traffic3 = load_model('traffic3' +'2L' + '.h5')



name = ['node1','node3','node5']#,'traffic1','traffic2','traffic3']
model = [model_node1,model_node3,model_node5]#,model_traffic1,model_traffic2,model_traffic3]
#第一轮
for i in range(len(name)):              
    dataset_node = []
    pattern = name[i]
    data = csvtolist(pattern)
    for k in range(2000):
        dataset_node.append(data[k][0])
    if(i <= 3):
        dataset_node = list(np.array(dataset_node)/300.0)
    else:
        dataset_node = list(np.array(dataset_node)/150000000.0)
    dataset_node = preprocess(dataset_node) 
    dataset_node_pre = [dataset_node[240:1440]]
    predict_data = predict(dataset_node_pre,i)
    dataset_node_pre = dataset_node[240:1440]
    #print(predict_data)v
    start = start_time(predict_data)
    start_set.append(start)
    sum_node = sum(predict_data)
    sum_node_set.append(sum_node)
    dataset_node_set.append( dataset_node_pre )
#建立socket通信168.109.124'


while(1):
     #载入模型
    start1 = time.time()
    print(start1)
    model_node1 = load_model('node1' +'2L' + '.h5')
    #model_node2 = load_model('node2' +'2L' + '.h5')
    model_node3 = load_model('node3' +'2L' + '.h5')
    #model_node4 = load_model('node4' +'2L' + '.h5')
    model_node5 = load_model('node5' +'2L' + '.h5')
    #model_node6 = load_model('node6' +'2L' + '.h5')
    #model_traffic1 = load_model('traffic1' +'2L' + '.h5')
    #model_traffic2 = load_model('traffic2' +'2L' + '.h5')
    #model_traffic3 = load_model('traffic3' +'2L' + '.h5')
    end1 = time.time()
    spendtime = end1 - start1
    model = [model_node1,model_node3,model_node5]#,model_traffic1,model_traffic2,model_traffic3]
#接受数据
    #流量采集的地址
    client,address=sock_server.accept()
    datarecv = client.recv(10086)
    client.close()
    datarecv = datarecv.decode('utf-8')
    print(datarecv)
    datarecv = json.loads(datarecv)
    print(type(datarecv))
    datarecv_set = []
    trainX_set = []
    trainY_set = []
    for i in range(len(name)):
        #print(type(datarecv[name[i]]))
        datarecv_set.append(list(np.array(datarecv[name[i]])/300.0))
    for i in range(len(name)):
         #print(type(datarecv_set[i]))
         datarecv_set[i] = preprocess1(datarecv_set[i])
         print('nodedata:', datarecv_set[i]  )
         # dataset_node_set[i] = list(dataset_node_set[i])
         print(dataset_node_set[i])
         for k in range(len(datarecv_set[i])):
             dataset_node_set[i].append(datarecv_set[i][k])
         trainX, trainY = creat_dataset1(dataset_node_set[i][-(look_back + 1* predictSampleNum):], look_back)  # 产生训练样本
         trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
         trainX_set.append(trainX)
         trainY_set.append(trainY)
         dataset_node_pre = [dataset_node_set[i][-(look_back):]]
         dataset_node_pre = np.array(dataset_node_pre)
         test_inputdata = np.reshape(dataset_node_pre, (dataset_node_pre.shape[0], 1, dataset_node_pre.shape[1]))
         for k in range(test_inputdata.shape[0]):
             test_temp = np.array([test_inputdata[k, :, :]])
             predict_node = model[i].predict(test_temp)
         predict_node = predict_node[0]
         start = start_time(predict_node)
         start_set.append(start)
         sum_node = sum(predict_node)
         sum_node_set.append(sum_node)
         # start = time.time()
         # model[i].fit(trainX, trainY, epochs=5, batch_size=32)
         # end = time.time()
         # spend_time = end - start
         # print('time', spend_time)
         # model[i].save(name[i] + '2L' + '.h5')

#发送数据
    #send_list={'node1':[start_set[0],sum_node_set[0]], 'node2':[start_set[1],sum_node_set[1]],\
               #'node3':[start_set[2],sum_node_set[2]], 'node4':[start_set[3],sum_node_set[3]],\
              # 'node5':[start_set[4],sum_node_set[4]], 'node6':[start_set[5],sum_node_set[5]]]#,\
               #'traffic1':[start_set[6],sum_node_set[6]], 'traffic2':[start_set[7],sum_node_set[7]], 'traffic3':[start_set[8],sum_node_set[8]]}
    send_list=[sum_node_set[0],sum_node_set[1],sum_node_set[2]]
    print('predicted data',send_list)
    datasend = json.dumps(send_list)
    # sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock1.connect(('192.168.108.222', 23455))
    # sock1.send(json.dumps(send_list).encode('utf-8'))
    # sock1.close()
    start_set=[]
    sum_node_set = []
    print("send success")


    # datarecv_set=[]
    # for i in name:
    #     datarecv_set.append(datarecv[name[i]])

    for i in range(len(name)):

        # datarecv_set[i] = preprocess(datarecv_set[i])
        # for k in range(len(datarecv_set[i])):
        #     dataset_node_set[i] =  dataset_node_set[i].append(datarecv_set[i][k])
        # trainX, trainY = creat_dataset(dataset_node_set[i][-(look_back+2*predictSampleNum):], look_back)  #产生训练样本
        # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        # dataset_node_pre = [dataset_node_set[i][-(look_back):]]
        # dataset_node_pre = np.array(dataset_node_pre)
        # test_inputdata = np.reshape(dataset_node_pre, (dataset_node_pre.shape[0], 1, dataset_node_pre.shape[1]))
        # for k in range(test_inputdata.shape[0]):
        #      test_temp = np.array([test_inputdata[k, :, :]])
        #      predict_node = model[i].predict(test_temp)
        # predict_node = predict_node[0]
        # start = start_time(predict_node)
        # start_set = start_set.append(start)
        # sum_node = sum(predict_node)
        # sum_node_set = sum_node_set.append(sum_node)
        #
        start = time.time()
        model[i].fit(trainX_set[i], trainY_set[i], epochs=1, batch_size=1)
        end =time.time()
        spend_time = end -start
        print('time',spend_time)
        model[i].save(name[i] + '2L'+ '.h5')

            
            
    
    
    