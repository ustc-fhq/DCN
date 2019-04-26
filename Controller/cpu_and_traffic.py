from influxdb import InfluxDBClient
import time
import json
import requests
from time import sleep
import datetime
import pandas as pd
import socket

'''
进行cpu和流量信息的收集，同时每隔一段时间就取出数据
'''


# 重整流量信息
def reshape(traffic_list):
    time_index = pd.to_datetime(traffic_list[0], unit='s')
    series = pd.Series(traffic_list[1], index=time_index)
    series = series.resample('5s').mean().ffill()
    a = list()
    a.append(float(series.sum()))
    for i in range(23):
        a.append(0.0)
    return a


# 收集cpu信息的函数,返回字典
def cpu_collect(*args):
    # 采集的host名字
    host = args
    now_time=int(time.time()*10**9)
    client = InfluxDBClient('192.168.108.217', 8086, 'root', 'root', 'collectd')
    cpu_dic = {}
    for i in range(len(host)):
        cpu_list=[]
        # 定义收集前几分钟数据以及采样间隔，只用改这里的参数
        sql_search = "SELECT mean(\"value\") AS \"mean_value\" FROM \"" \
                     "collectd\".\"rp_day_1\".\"cpu_value\" WHERE " \
                     "time >now()-118s AND \"host\"='"+host[i]+"' " \
                     "AND \"type_instance\"='user' GROUP BY time(5s)"
        result=client.query(sql_search)
        result=result.raw
        for j in range(24):
            if type(result['series'][0]['values'][j][1]) != float:
                cpu_list.append(0)
            else:
                cpu_list.append(result['series'][0]['values'][j][1])
        cpu_dic[host[i]]=cpu_list
    client.close()
    return cpu_dic


def send_information():
    vm1 = ['10.10.10.27', '10.10.10.19']
    vm2 = ['10.10.10.20', '10.10.10.23']
    vm3 = ['10.10.10.22', '10.10.10.5']
    vm_cluster = [vm1, vm2, vm3]
    # 获取时间，如果前后两个时间超过预测周期，如两分钟，则把字典socket过去
    for i in range(len(vm_cluster)):
        flow1 = {'keys': 'ipsource,ipdestination', 'value': 'bytes',
                 'filter': 'ipsource=' + vm_cluster[i][0] + '&ipdestination=' + vm_cluster[i][1]}
        flow2 = {'keys': 'ipsource,ipdestination', 'value': 'bytes',
                 'filter': 'ipsource=' + vm_cluster[i][1] + '&ipdestination=' + vm_cluster[i][0]}
        requests.put('http://192.168.108.222:8008/flow/' + str(i) + str(0) + '/json', data=json.dumps(flow1))
        requests.put('http://192.168.108.222:8008/flow/' + str(i) + str(1) + '/json', data=json.dumps(flow2))

    traffic_1 = [[], []]
    traffic_2 = [[], []]
    traffic_3 = [[], []]
    traffic_sum = [traffic_1, traffic_2, traffic_3]
    # 获取时间，如果前后两个时间超过预测周期，如两分钟，则把字典socket过去
    now_time = time.time()
    time1 = now_time
    while 1:
        now_time = time.time()
        time2 = now_time
        for i in range(3):
            r1 = requests.get('http://192.168.108.222:8008/activeflows/ALL/' + str(i) + str(0) + '/json')
            r2 = requests.get('http://192.168.108.222:8008/activeflows/ALL/' + str(i) + str(1) + '/json')
            traffic_now1 = r1.json()
            if (len(traffic_now1) == 0):
                traffic1 = 0
            else:
                traffic1 = int(traffic_now1[0]['value'])
            traffic_now2 = r2.json()
            if (len(traffic_now2) == 0):
                traffic2 = 0
            else:
                traffic2 = int(traffic_now2[0]['value'])
            traffic = traffic1 + traffic2
            traffic_sum[i][0].append(now_time)
            traffic_sum[i][1].append(traffic)
        if (time2 - time1) >= 11:
            time1 = time2
            sock_dir = cpu_collect()
            sock_dir['trattfic0'] = reshape(traffic_sum[0])
            sock_dir['trattfic1'] = reshape(traffic_sum[1])
            sock_dir['trattfic2'] = reshape(traffic_sum[2])
            traffic_1 = [[], []]
            traffic_2 = [[], []]
            traffic_3 = [[], []]
            traffic_sum = [traffic_1, traffic_2, traffic_3]
            print(sock_dir)
            needsend = json.dumps(sock_dir)

