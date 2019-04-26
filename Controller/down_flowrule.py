import socket
import json
import socket
from vm_migrate import vm_migrate
def down_flowrule(i,flag):
    host=['192.168.108.218','192.168.108.219','192.168.108.221']
    vm=["10.10.10.11","10.10.10.24","10.10.10.17","10.10.10.15","10.10.10.20","10.10.10.9","10.10.10.7","10.10.10.22","10.10.10.5"]
    vm_cluster=[[],[],[]]
    vm_cluster[0]=vm[0:3]
    vm_cluster[1]=vm[3:6]
    vm_cluster[2]=vm[6:9]
    flow_info=vm_cluster[i][1]+':'+vm_cluster[i][2]+':'+str(flag)+'\n'
    print(flow_info)
    return flow_info
