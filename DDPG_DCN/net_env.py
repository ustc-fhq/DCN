import numpy as np
import copy


'''
DCN environment
'''
class DCNEnv(object):
    def __init__(self, rack_num, ):
        self._rack_num = rack_num



class Rack(object):
    def __init__(self):
        pass

def main():
    pass


if __name__ == '__main__':
    main()

'''
vm_cluster=3
vm_info_out=np.zeros((vm_cluster,2,2), dtype=np.int)
cpu_level_num_out = 5
cpu_threshold_out = 6
server_num_out = 3
port_num_out = 4
traffic_threshold_out = np.array([10, 0.05, 5, 5])
vm_pair_out=np.zeros((3,2),dtype=np.int)
network_num_out = 4
network_info_out=0
state_dim_out = 19
action_dim_out = 6
constant_parameter_set = np.array([1,1,1])
config_port_num = np.zeros((network_num_out, server_num_out, server_num_out))

vm_location_bound = np.array([2,2,2,2,2,2,2,2,2])
vm_pair_traffic_network_bound = np.array([3,3,3,3,3,3,3,3,3])
network_reconfiguration_bound = np.array([3])
action_bound_out = np.array([5,2,3,3,3,3])
config_port_num = np.zeros((network_num_out, server_num_out, server_num_out))

class NetEnv(object):
        state_dim=state_dim_out
        action_dim=action_dim_out
        server_num=server_num_out
        action_bound=action_bound_out
        config_port_num=config_port_num
        port_num=port_num_out
        network_num=network_num_out
        def __init__(self):
            self.vm_info=vm_info_out
            self.vm_pair=vm_pair_out
            self.network_info=network_info_out

        def get_state(self):
            state=np.array([])
            for i in range(len(self.vm_info)):
                for j in range(len(self.vm_info[i])):
                    state=np.append(self.vm_info[i][j][0])
                    state=np.append(self.vm_info[i][j][1])
            return state

        def state_vm(self,state):
            state_count=0
            for i in range(len(self.vm_info)):
                for j in range(len(self.vm_info[i])):
                    self.vm_info[i][j][0]=state[state_count]
                    state_count+=1
                    self.vm_info[i][j][1]=state[state_count]
                    state_count+=1

            for i in range(3):
                self.vm_pair[i][0]=state[state_count]
                state_count+=1
                self.vm_pair[i][1]=state[state_count]
                state_count+=1
            self.network_info=state[state_count]
        #send legal action
        def action_vm(self,action):
            reward=0
            vm_info_new=copy.deepcopy(self.vm_info)
            vm_pair_new=copy.deepcopy(self.vm_pair)
            num=action[0]%2
            host_vm=[0,0,0]
            if action[1]!=0:
                reward=-1
                #num=action[0]%3
                if action[1]==1:
                    vm_info_new[int((action[0]-num)/2)][int(num)][0]-=1
                    if vm_info_new[int((action[0]-num)/2)][int(num)][0]<0:
                        vm_info_new[int((action[0]-num)/2)][int(num)][0]+=3
                else:
                    vm_info_new[int((action[0]-num)/2)][int(num)][0]+=1
                    if vm_info_new[int((action[0]-num)/2)][int(num)][0]>2:
                        vm_info_new[int((action[0]-num)/2)][int(num)][0]-=3
            for i in range(3):
                for j in range(2):
                    host_vm[vm_info_new[i][j][0]]+=1
            if host_vm[action[1]]>5:
                reward=0
                vm_info_new=copy.deepcopy(self.vm_info)
            network_num=action[-1]
            for i in range(3):
                if vm_info_new[i][0][0]==vm_info_new[i][0][1]:
                    vm_pair_new[i][0]=0
                elif config_port_num[network_num][vm_info_new[i][0][0]][vm_info_new[i][1][0]]==0:
                    vm_pair_new[i][0]=1
                elif config_port_num[network_num][vm_info_new[i][0][0]][vm_info_new[i][1][0]]==1:
                    if action[i+2]>2:
                        vm_pair_new[i][0]=2
                else:
                    vm_pair_new[i][0]=action[i+2]
            return reward,vm_info_new,vm_pair_new

        #send legal action
        def reward_calculation(self, action):
            reward_cal=0
            reward,vm_info_new,vm_pair_new=self.action_vm(action)
            reward_cal+=reward
            #current_node_congestion_set = np.zeros(self.server_num)
            #updated_node_congestion_set = np.zeros(self.server_num)
            #current_link_congestion_set = np.zeros((3,4))
            #updated_link_congestion_set = np.zeros((3,4))
            current_accumulative_cpu_level = [0,0,0]
            updated_accumulative_cpu_level = [0,0,0]
            current_link_capacity=np.zeros((3,4))
            updated_link_capacity=np.zeros((3,4))
            for i in range(len(self.vm_info)):
                for j in range(len(self.vm_info[i])):
                    current_accumulative_cpu_level[int(self.vm_info[i][j][0])]+=self.vm_info[i][j][1]
                    updated_accumulative_cpu_level[int(vm_info_new[i][j][0])]+=vm_info_new[i][j][1]
            for i in range(len(self.vm_pair)):
                        current_link_capacity[i][self.vm_pair[i][0]]+=self.vm_pair[i][1]
                        updated_link_capacity[i][vm_pair_new[i][0]]+=vm_pair_new[i][1]
            current_reward_cal=0
            update_reward_cal=0
            for i in range(self.server_num):
                if current_accumulative_cpu_level[i]>cpu_threshold_out:
                    current_reward_cal+=(current_accumulative_cpu_level[i]-cpu_threshold_out)
                if updated_accumulative_cpu_level[i]>cpu_threshold_out:
                    update_reward_cal+=(updated_accumulative_cpu_level[i]-cpu_threshold_out)
            for i in range(len(self.vm_pair)):
                for j in range(4):
                    if current_link_capacity[i][j]>traffic_threshold_out[j]:
                        current_reward_cal+=(current_link_capacity[i][j]-traffic_threshold_out[j])
                    if updated_link_capacity[i][j]>traffic_threshold_out[j]:
                        update_reward_cal+=(updated_link_capacity[i][j]-traffic_threshold_out[j])
            reward_cal+=(0.5*current_reward_cal-update_reward_cal+1.5)
            return reward_cal*100
'''





















