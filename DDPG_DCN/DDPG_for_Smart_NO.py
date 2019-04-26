import tensorflow as tf
import numpy as np
import os
import shutil
from net_env import NetEnv
import socket
import multiprocessing
import json
import time
import pickle
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
np.random.seed(1)
tf.set_random_seed(1)

START_TRAIN=0
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0  # reward discount
MEMORY_CAPACITY = 10000
BATCH_SIZE = 20
VAR_MIN = 0.05
VAR1_MIN=0.2
LOAD = False
REPLACE_ITER_A = 100
REPLACE_ITER_C = 100

env = NetEnv()  # load network
eval_net=NetEnv()
STATE_DIM = env.state_dim  # load state dimension 
ACTION_DIM = env.action_dim  # load action dimension
ACTION_BOUND = env.action_bound
SERVER_NUM = env.server_num


# all placeholder for tf
with tf.name_scope('S'):  # current state
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):  # reward
    R = tf.placeholder(tf.float32, [None, 1], name='r')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate,t_replace_iter):
        self.sess = sess  # initial a section
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter=t_replace_iter
        self.t_replace_counter=0

        with tf.variable_scope('Actor'):
            # input s, output a, build an evaluation network
            self.a = self._build_net(S, scope='eval_net', trainable=True)
            #self.a_ = self._build_net(S, scope='target_net', trainable=False)
        # get parameters in evaluation and target networks
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        #self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()   # initialize parameters 
            init_b = tf.constant_initializer(0.001)           # initialize parameter bias
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, self.a_dim, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf. layers.dense(net, self.a_dim, activation=tf.nn.sigmoid, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        #if self.t_replace_counter % self.t_replace_iter == 0:
            #self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        #self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))



class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a):#, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)
            #self.q_ = self._build_net(S, a_, 'target_net', trainable=False)
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            #self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R#+ self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
        return q

    def learn(self, s, a, r):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r})
        #if self.t_replace_counter % self.t_replace_iter == 0:
            #self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        #self.t_replace_counter += 1

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r):
        transition = np.hstack((s, a, [r]))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND, LR_A,REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA,REPLACE_ITER_C ,actor.a)#,actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=STATE_DIM + ACTION_DIM + 1)
tf.summary.FileWriter("logs/",sess.graph)
sess.run(tf.global_variables_initializer())
State_store={'number':0}
saver=tf.train.Saver()

def train(var,var1,START_TRAIN):

      # control exploration
    for i in range(1000):
        if State_store['number'] >0:
            random_int=np.random.randint(1,State_store['number']+1)
            s=State_store[str(random_int)]
            a=actor.choose_action(s)
            if(var>np.random.rand(1)):
                a[0]=np.random.randint(0,6)
                a[1]=np.random.randint(0,3)
            a=a.astype(np.int)
            if(a[0]>5):
                a[0]=5
            elif(a[0]<0):
                a[0]=0
            if(a[1]>2):
                a[1]=2
            elif(a[1]<0):
                a[1]=0
            env.state_vm(s)
            #print(a)
            r=env.reward_calculation(a)
            #print(r)
            M.store_transition(s,a,r)
    #print(M.pointer)
    if M.pointer > MEMORY_CAPACITY:
        START_TRAIN=1
        var-=0.05
        for k in range(5000):
                #print(START_TRAIN)
                #var = max([var*.99999, VAR_MIN])
                #var1 = max([var1*.99999, VAR_MIN])# decay the action randomness
                b_m = M.sample(BATCH_SIZE)
                b_s = b_m[:, :STATE_DIM]
                b_a = b_m[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_m[:, -1: ]
                #print(b_m)
                #print(b_s)
                #print(b_a)
                #print(b_r)

                critic.learn(b_s, b_a, b_r)
                actor.learn(b_s)
    return var,var1,START_TRAIN

def constrained_action(a):
    vm=a[0:9]
    vm=vm.astype(int)
    vm_pair=np.ceil(a[9:18])
    network_configuration_a = a[-1]
    network_configuration_a=network_configuration_a.astype(int)
    for i in range(3):
        for j in range(3):
            for k in range(i+1,3):
                if(vm[3*i+j]==vm[3*i+k]):
                    vm_pair[3*i+j+k]=0
    constrained_a = np.hstack((vm,vm_pair,network_configuration_a))
    return constrained_a


def evaluate(s):
    a = actor.choose_action(s)
    #constrained_a = constrained_action(a)
    #eval_net.state_vm(s)
    #r = eval_net.reward_calculation(constrained_a)
    return a#constrained_a#, r





if __name__ == '__main__':
    '''
    var=1
    var1=3
    while 1:
        is_file=os.path.isfile('state.pkl')
        if is_file:
            time.sleep(1)
            f=open('state.pkl','rb')
            state=pickle.load(f)
            f.close()
            state_number=State_store['number']
            State_store[str(state_number+1)]=state
            State_store['number']=state_number+1
            action=evaluate(state)

            env.state_vm(state)
            r=env.reward_calculation(action)
            print(r)

            time.sleep(1)
            sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            sock.connect(('127.0.0.1',23345))
            sock.send(json.dumps(action.tolist()).encode('utf-8'))
        else:
            var,var1,START_TRAIN=train(var,var1,START_TRAIN)
    '''
    var=1
    var1=3
    vm_station=[0,1,1,2,2,0,1,1,1,0]
    ip_info1=[5,5,0,0,5,5,0,5,0]
    ip_info2=[0,0,5,5,5,5,5,0,0]
    ip_info3=[5,5,0,0,0,0,0,5,5]
    ip_info=[ip_info1,ip_info2,ip_info3]
    i=0
    k=0
    k+=1
    if k==12:
        vm_station=[0,1,1,2,2,0,1,1,1]
        k-=12
    state=[]
    if i>2:
        i-=3
    for j in range(9):
        state.append(vm_station[j])
        state.append(ip_info[i][j])
    state.append(vm_station[-1])
    i+=1
    state=np.array(state)
    state_number=State_store['number']
    State_store[str(state_number+1)]=state
    State_store['number']=state_number+1
    #print(state)

    action=actor.choose_action(state)
    action=action.astype(np.int)
    if(action[0]>5):
        action[0]=5
    elif(action[0]<0):
        action[0]=0
    if(action[1]>2):
        action[1]=2
    elif(action[1]<0):
        action[1]=0
    for traffic in range(4):
        if(action[traffic+2]>3):
            action[traffic+2]=3
        elif(action[traffic+2]<0):
            action[traffic+2]=0
    print(state)
    print(action)
    env.state_vm(state)
    r=env.reward_calculation(action)
    print(r)
    r1,vm_info_new,vm_pair_new=env.action_vm(action)
    n=0
    for l in range(3):
        for m in range(2):
            vm_station[n]=vm_info_new[l][m][0]
            n+=1
    for l in range(3):
        vm_station[n]=vm_pair_new[l][0]
    while 1:
        if START_TRAIN==1:

            if k==3:
                vm_station=[0,1,1,2,2,0,1,1,1,0]
                k-=3
            k+=1
            state=[]
            if i>2:
                i-=3
            for j in range(9):
                state.append(vm_station[j])
                state.append(ip_info[i][j])
            state.append(vm_station[-1])
            i+=1
            state=np.array(state)
            state_number=State_store['number']
            State_store[str(state_number+1)]=state
            State_store['number']=state_number+1
            #print(state)

            action=actor.choose_action(state)
            action=action.astype(np.int)
            if(action[0]>5):
                action[0]=5
            elif(action[0]<0):
                action[0]=0
            if(action[1]>2):
                action[1]=2
            elif(action[1]<0):
                action[1]=0
            for traffic in range(4):
                if(action[traffic+2]>3):
                    action[traffic+2]=3
                elif(action[traffic+2]<0):
                    action[traffic+2]=0
            print(state)
            print(action)
            env.state_vm(state)
            r=env.reward_calculation(action)
            print(r)
            r1,vm_info_new,vm_pair_new=env.action_vm(action)
            n=0
            for l in range(3):
                for m in range(2):
                    vm_station[n]=vm_info_new[l][m][0]
                    n+=1
            for l in range(3):
                vm_station[n]=vm_pair_new[l][0]

            #print(State_store)
            #print(var)
            print(vm_station)
        var,var1,START_TRAIN=train(var,var1,START_TRAIN)
        print(var)
        if var<=0.3:
            saver.save(sess,'./models.ckp',1000)
        #'''









