import threading
import sys
from Controller import start_task
from Controller import cpu_and_traffic
from DDPG_DCN import DDPG_for_Smart_NO
from DDPG_DCN import net_env

t1 = threading.Thread(target=start_task.start_task)
t2 = threading.Thread(target=cpu_and_traffic.send_information())