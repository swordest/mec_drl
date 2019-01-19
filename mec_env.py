import numpy as np
from helper import *
from ddpg_lib import * 
import ipdb as pdb

class MecTerm(object):
    """
    MEC terminal parent class
    """
    def __init__(self, user_config, train_config):
        self.rate = user_config['rate']
        self.dis = user_config['dis']
        self.id = user_config['id']
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.action_bound = user_config['action_bound']
        self.data_buf_size = user_config['data_buf_size']
        self.t_factor = user_config['t_factor']
        self.penalty = user_config['penalty']

        self.channelModel = MarkovModel(self.dis)

        self.DataBuf = 0
        self.Channel = self.channelModel.getCh()
        self.SINR = 0
        self.Power = np.zeros(2)
        self.Reward = 0
        self.State = []
        
        # some pre-defined parameters
        self.k = 1e-27
        self.t = 0.001
        self.L = 500
        self.sigma2 = 0.3e-9
    
    def localProc(self, p):
        return np.power(p/self.k, 1.0/3.0)*self.t/self.L/1000
    
    def localProcRev(self, b):
        return np.power(b*1000*self.L/self.t, 3.0)*self.k
    
    def offloadRev(self, b):
        return (np.power(2.0, b)-1)*self.sigma2/np.power(self.Channel, 2.0)
    
    def getCh(self):
        return self.Channel
    
    def setSINR(self, sinr):
        self.SINR = sinr
        self.sampleCh()
        self.State = np.array([self.DataBuf, self.SINR, self.Channel])
        
class MecTermGD(MecTerm):
    """
    MEC terminal class using Greedy algorithms
    """
    
    def __init__(self, user_config, train_config, policy, data_rec):
        MecTerm.__init__(self, user_config, train_config)
        self.policy = policy # 
        self.res_rec = data_rec[0] # channel and data arrival for each slot
        self.init_buf = data_rec[1]
        self.seq_index = 0
        self.frame_index = -1
        
        self.local_proc_max_bits = self.localProc(self.action_bound) # max processed bits per slot
        
    def feedback(self, sinr, done):
        self.SINR = sinr

        # update the data buffer
        [data_t, data_p, data_r] = self.sampleData()
        
        self.seq_index += 1
        self.sampleCh()
        
        return self.Reward, np.sum(self.Power), data_t, data_p, data_r, self.DataBuf, self.Channel
    
    def predict(self):
        data = self.DataBuf
        if self.policy == 'local':
            self.offloadDo(self.localProcDo(data))
        else: 
            self.localProcDo(self.offloadDo(data))
        
        self.Power = np.fmax(0, np.fmin(self.action_bound, self.Power))
        return self.Power, np.zeros([self.action_dim])
    
    def localProcDo(self, data):
        if self.local_proc_max_bits < data:
            self.Power[1] = 1.0
            data -= self.local_proc_max_bits
        else:
            self.Power[1] = self.localProcRev(data)
            data = 0
        return data
    
    def offloadDo(self, data):
        offload_max_bits = np.log2(1+np.power(self.Channel, 2.0)/self.sigma2)
        if offload_max_bits < data:
            self.Power[0] = 1.0
            data -= offload_max_bits
        else:
            self.Power[0] = self.offloadRev(data)
            data = 0
        return data

    def reset(self):        
        self.seq_index = 0
        self.frame_index += 1
        self.DataBuf = self.init_buf[self.frame_index][0]
        return self.DataBuf
    
    def sampleCh(self):
        if self.seq_index == 0:
            self.Channel = self.res_rec[self.frame_index][0][8]
        else:
            self.Channel = self.res_rec[self.frame_index][self.seq_index-1][8]
        return self.Channel
    
    def sampleData(self):
        data_t = np.log2(1 + self.SINR)
        data_p = self.localProc(self.Power[1])
        data_r = self.res_rec[self.frame_index][self.seq_index][6][0]
        self.DataBuf = np.fmin(self.data_buf_size, np.fmax(0, self.DataBuf-data_t-data_p)+data_r)
        return data_t, data_p, data_r

class MecTermRL(MecTerm):
    """
    MEC terminal class using RL
    """

    # rate:packet poisson arrival, dis: distance in meters
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        self.agent = DDPGAgent(sess, self.state_dim, self.action_dim,
                                self.action_bound, train_config)

    def feedback(self, sinr, done):
        self.SINR = sinr

        # update the data buffer
        [data_t, data_p, data_r] = self.sampleData()

        # get the reward for the current slot
        self.Reward = -self.t_factor*np.sum(self.Power) - (1-self.t_factor)*self.DataBuf

        # penalty for data buffer overflow
        if self.DataBuf >= self.data_buf_size:
            self.Reward -= self.penalty

        # should ignore some starting steps? consider it next 

        # estimate the channel for next slot
        self.sampleCh()

        # update the actor and critic network
        next_state = np.array([self.DataBuf, self.SINR, self.Channel])
        self.agent.update(self.State, self.Power, self.Reward, done, next_state)

        # update system state
        self.State = next_state
        # return the reward in this slot
        return self.Reward, np.sum(self.Power), data_t, data_p, data_r, self.DataBuf, self.Channel

    def reset(self):
        self.DataBuf = np.random.randint(0, self.data_buf_size-1)
        self.sampleCh()
        return self.DataBuf

    def sampleCh(self):
        self.Channel = self.channelModel.sampleCh()
        return self.Channel

    def sampleData(self):
        data_t = np.log2(1 + self.SINR)
        data_p = self.localProc(self.Power[1])
        data_r = np.random.exponential(self.rate)
        self.DataBuf = np.fmin(self.data_buf_size, np.fmax(0, self.DataBuf-data_t-data_p)+data_r)
        return data_t, data_p, data_r

    def predict(self):
        power, noise = self.agent.predict(self.State)
        self.Power = np.fmax(0, np.fmin(self.action_bound, power))
        return self.Power, noise


class MecSvrEnv(object):
    """
    Simulation environment
    """
    def __init__(self, user_list, num_att, sigma2, max_len): 
        self.user_list = user_list
        self.num_user = len(user_list)
        self.num_att = num_att
        self.sigma2 = sigma2
        self.count = 0
        self.max_len = max_len
        
    def init_target_network(self):
        for user in self.user_list:
            user.agent.init_target_network()

    def step_transmit(self):
        # get the channel vectors 
        channels = [user.getCh() for user in self.user_list]
        # get the transmit powers
        powers = []
        noises = []
        for i in range(self.num_user):
            p, n = self.user_list[i].predict()
            powers.append(p)
            noises.append(n)
        # compute the sinr for each user
        
        powers = np.array(powers)
        noises = np.array(noises)
        sinr_list = self.compute_sinr(channels, powers[:,0])

        rewards = np.zeros(self.num_user)
        powers = np.zeros(self.num_user)
        data_ts = np.zeros(self.num_user)
        data_ps = np.zeros(self.num_user)
        data_rs = np.zeros(self.num_user)
        data_buf_sizes = np.zeros(self.num_user)
        next_channels = np.zeros(self.num_user)
        
        self.count += 1
        # feedback the sinr to each user
        for i in range(self.num_user):
            [rewards[i], powers[i], data_ts[i], data_ps[i], data_rs[i], data_buf_sizes[i], next_channels[i]] = self.user_list[i].feedback(sinr_list[i], self.count >= self.max_len)
            
        return rewards, self.count >= self.max_len, powers, noises, data_ts, data_ps, data_rs, data_buf_sizes, next_channels        

    def compute_sinr(self, channels, powers):
        # calculate the received power at the MEC server for each user
        channel_gains = np.sum(np.power(np.abs(channels), 2), axis=1)
        receive_powers = channel_gains*powers
        total_power = np.sum(receive_powers)

        # ordering the channels by their power gain in an acending order
        idx_list = np.argsort(receive_powers)[::-1]

        # get access to the channel and decode in an decending order
        sinr_list = np.zeros(self.num_user)
        total_power -= receive_powers[idx_list[0]]
        for i in range(self.num_user):
            user_idx = idx_list[i]
            sinr_list[user_idx] = receive_powers[user_idx]/(total_power+self.sigma2)
            total_power -= receive_powers[user_idx]
        return sinr_list

    def reset(self):
        self.count = 0
        init_data_buf_size = [user.reset() for user in self.user_list]

        # get the channel vectors 
        channels = [user.getCh() for user in self.user_list]
        # get the transmit powers
        powers = [np.random.uniform(0, user.action_bound) for user in self.user_list]
        # compute the sinr for each user
        sinr_list = self.compute_sinr(channels, powers)

        for i in range(self.num_user):
            self.user_list[i].setSINR(sinr_list[i])
        return init_data_buf_size

