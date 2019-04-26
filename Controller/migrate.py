import subprocess
import oxc
import socket
import json
#虚拟机迁移
from vm_migrate import vm_migrate
#下发流表
from down_flowrule import down_flowrule
import pickle
import time
import os
import numpy as np
from sendflowrule import sendflowrule
is_file=1
host=['192.168.108.218','192.168.108.219','192.168.108.221']
#连接onos下发流表
sock_lqhz=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock_lqhz.connect(('192.168.109.123',14000))
#连接AI模块
sock_AI=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock_AI.connect(('192.168.108.222',52131))
#intinitional state
#每次初始化查一下
vm_station=[0,0,0,1,1,1,2,2,2]
oxc_state=0
#连接onos发刘表
sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.bind(('192.168.108.222',23455))
sock.listen(5)
while 1:
    #get traffic predicted and CPU prediction,connect to traffic
    client1,addr1=sock.accept()
    cpu_traffic=client1.recv(1024)
    client1.close()
    cpu_traffic=json.loads(cpu_traffic.decode('utf-8'))
    #当前状态,包含虚拟机位置,cpu和流量信息,光网络配置
    state=[]
    state.append(vm_station)
    state.append(cpu_traffic)
    state.append(oxc_state)
    #把网络配置发送给AI模块
    sock_AI.send(json.dumps(state).encode('utf-8'))
    action=sock_AI.recv(1024)
    action=json.loads(action.decode('utf-8'))
    print(action)
    if action[1]==1:
        host_num=vm_station[action[0]]-1
        if host_num<0:
            host_num+=3
        vm_migrate(action[0],vm_station[action],host_num)
        vm_station[action[0]]=host_num
    elif action[1]==2:
        host_num=vm_station[action[0]]-1
        if host_num>2:
            host_num-=3
        vm_migrate(action[0],vm_station[action[0]],host_num)
        vm_station[action[0]]=host_num
    sock_lqhz.send(down_flowrule(0,action[2]).encode('utf-8'))
    sock_lqhz.send(down_flowrule(1,action[3]).encode('utf-8'))
    sock_lqhz.send(down_flowrule(2,action[4]).encode('utf-8'))

