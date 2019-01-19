import os
import numpy as np
import tensorflow as tf
from ddpg_lib import * 
import ipdb as pdb
import matplotlib.pyplot as plt

state_avg = np.array([0.031,0.153,0.399,0.772,1.274,1.911,2.694,3.630,4.730,6.021,7.902])
trans_p = np.array([[0.514,0.514,1.000,],
[0.513,0.696,1.000,],
[0.513,0.745,1.000,],
[0.515,0.776,1.000,],
[0.513,0.799,1.000,],
[0.514,0.821,1.000,],
[0.516,0.842,1.000,],
[0.511,0.858,1.000,],
[0.516,0.880,1.000,],
[0.512,0.897,1.000,],
[0.671,1.000,1.000,],])   # stay, lower, higher
alpha = 3.0
ref_loss = 0.001

class MarkovModel(object):
    """docstring for MarkovModel"""
    def __init__(self, dis, seed=123):
        # self.f_m = f_m
        # self.t_p = t_p
        self.dis = dis
        self.path_loss = ref_loss*np.power(1./dis, alpha)
        np.random.seed([seed])

        # calculate the transition prob
        self.trans_p = trans_p
        self.state_avg = state_avg
        self.state = np.random.randint(0, 11) # states in {0,...,9}

    def getCh(self):
        return np.array([np.sqrt(self.path_loss*self.state_avg[self.state])])
    
    def sampleCh(self):
        temp = np.random.random()
        if temp >= trans_p[self.state, 1]:
            self.state += 1
        elif temp >= trans_p[self.state, 0]:
            self.state -= 1

        self.state = np.fmax(np.fmin(self.state, 11), 0)
        return self.getCh()

def complexGaussian(row=1, col=1, amp=1.0):
    real = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
    img = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
    return amp*(real + 1j*img)
    
class ARModel(object):
    """docstring for AR channel Model"""
    def __init__(self, dis, n_t=1, n_r=1, rho=0.95, seed=123):
        self.dis = dis
        self.n_t = n_t
        self.n_r = n_r
        self.path_loss = ref_loss*np.power(1./dis, alpha)
        np.random.seed([seed])
        
        self.rho = rho
        self.H = complexGaussian(self.n_t, self.n_r)  
        
    def getCh(self):
        return self.H*np.sqrt(self.path_loss)
        
    def sampleCh(self):
        self.H = self.rho*self.H + complexGaussian(self.n_t, self.n_r, np.sqrt(1-self.rho*self.rho))
        return self.getCh()
    
class DDPGAgentLD(object):
    """agent initialization from saved model"""
    def __init__(self, sess, user_config):
        self.sess = sess
        self.user_id = user_config['id']
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        
        # restore the input and output for the actor network
        self.actor = ActorNetworkLD(sess, self.user_id)
    
    def predict(self, s):
#         pdb.set_trace()
        return self.actor.predict(np.reshape(s, (1, self.state_dim)))[0]
    
    def init_target_network(self):
        pass
    
class DQNAgent(object):
    """docstring for DQNAgent"""
    def __init__(self, sess, user_config, train_config):
        self.sess = sess
        self.user_id = user_config['id']
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.action_bound = user_config['action_bound']
        self.action_level = user_config['action_level']
        self.minibatch_size = int(train_config['minibatch_size'])
        self.epsilon = float(train_config['epsilon'])
        
        self.action_nums = 1
        for i in range(self.action_dim):
            self.action_nums *= self.action_level
        
        self.max_step = 100000
        self.pre_train_steps = 5000
        self.total_step = 0
        self.DQN = DeepQNetwork(sess, self.state_dim, self.action_nums, float(train_config['critic_lr']), float(train_config['tau']), float(train_config['gamma']), self.user_id)
        
        self.replay_buffer = ReplayBuffer(int(train_config['buffer_size']), int(train_config['random_seed']))

    def init_target_network(self):
        self.DQN.update_target_network()
        
    def predict(self, s):
        if self.total_step <= self.max_step:
            self.epsilon *= 0.999976
            
        if np.random.rand(1) < self.epsilon or self.total_step < self.pre_train_steps:
            action = np.random.randint(0, self.action_nums)
        else:
            action, _ = self.DQN.predict(np.reshape(s, (1, self.state_dim)))
        
        self.total_step += 1
        return action, np.zeros([1])
    
    def update(self, s, a, r, t, s2):
        self.replay_buffer.add(np.reshape(s, (self.state_dim,)), a, r,
                              t, np.reshape(s2, (self.state_dim,)))
        
        if self.replay_buffer.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    self.replay_buffer.sample_batch(self.minibatch_size)

            # calculate targets
            _, q_out = self.DQN.predict(s_batch)
            target_prediction, target_q_out = self.DQN.predict_target(s2_batch)
            
            for k in range(self.minibatch_size):
                if t_batch[k]:
                    q_out[k][a_batch[k]] = r_batch[k]
                else:
                    q_out[k][a_batch[k]] = r_batch[k] + self.DQN.gamma * target_q_out[k][target_prediction[k]]

            # Update the critic given the targets
            q_loss, _ = self.DQN.train(
                s_batch, q_out) 
            
#             losses.append(q_loss)

            # Update target networks
            self.DQN.update_target_network()


class DDPGAgent(object):
    """docstring for DDPGAgent"""
    def __init__(self, sess, user_config, train_config):
        self.sess = sess
        self.user_id = user_config['id']
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.action_bound = user_config['action_bound']
        self.init_path = user_config['init_path'] if 'init_path' in user_config else ''
        
        self.minibatch_size = int(train_config['minibatch_size'])
        self.noise_sigma = float(train_config['noise_sigma'])

        # initalize the required modules: actor, critic and replaybuffer
        self.actor = ActorNetwork(sess, self.state_dim, self.action_dim, self.action_bound, float(train_config['actor_lr']), float(train_config['tau']), self.minibatch_size, self.user_id)
        
        self.critic = CriticNetwork(sess, self.state_dim, self.action_dim, float(train_config['critic_lr']), float(train_config['tau']), float(train_config['gamma']), self.actor.get_num_trainable_vars())
        
        self.replay_buffer = ReplayBuffer(int(train_config['buffer_size']), int(train_config['random_seed']))

        # mu, sigma=0.12, theta=.15, dt=1e-2,
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim),sigma=self.noise_sigma) 
#         self.actor_noise = GaussianNoise(0.1, 0.01, size=np.array([self.action_dim]))
        
    def init_target_network(self):
        # Initialize the original network and target network with pre-trained model
        if len(self.init_path) == 0:
            self.actor.update_target_network()
        else:
            self.actor.init_target_network(self.init_path)
        self.critic.update_target_network()

    # input current state and then return the next action
    def predict(self, s, isUpdateActor):
        if isUpdateActor:
            noise = self.actor_noise()
        else:
            noise = np.zeros(self.action_dim)
            
        return self.actor.predict(np.reshape(s, (1, self.actor.s_dim)))[0] + noise, noise
#         return self.actor.predict(np.reshape(s, (1, self.actor.s_dim))) + np.random.normal(0.0,0.1,[self.action_dim])

    def update(self, s, a, r, t, s2, isUpdateActor):
        self.replay_buffer.add(np.reshape(s, (self.actor.s_dim,)), np.reshape(a, (self.actor.a_dim,)), r,
                              t, np.reshape(s2, (self.actor.s_dim,)))
        
        if self.replay_buffer.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    self.replay_buffer.sample_batch(self.minibatch_size)

            # calculate targets
            target_q = self.critic.predict_target(
                    s2_batch, self.actor.predict_target(s2_batch))

            y_i = []
            for k in range(self.minibatch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _ = self.critic.train(
                s_batch, a_batch, np.reshape(y_i, (self.minibatch_size, 1)))

            if isUpdateActor:
                # Update the actor policy using the sampled gradient
                a_outs = self.actor.predict(s_batch)
                grads = self.critic.action_gradients(s_batch, a_outs)
                self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()
            
            
def test_helper(env, num_steps):
    cur_init_ds_ep = env.reset()
    
    user_list = env.user_list
    cur_r_ep = np.zeros(len(user_list))
    cur_p_ep = np.zeros(len(user_list))
    cur_ts_ep = np.zeros(len(user_list))
    cur_ps_ep = np.zeros(len(user_list))
    cur_rs_ep = np.zeros(len(user_list))
    cur_ds_ep = np.zeros(len(user_list))
    cur_ch_ep = np.zeros(len(user_list))
    
    for j in range(num_steps):
        # first try to transmit from current state
        [cur_r, done, cur_p, cur_n, cur_ts, cur_ps, cur_rs, cur_ds, cur_ch] = env.step_transmit()
        
        cur_r_ep += cur_r
        cur_p_ep += cur_p
        cur_ts_ep += cur_ts
        cur_rs_ep += cur_rs
        cur_ds_ep += cur_ds
        cur_ch_ep += cur_ch        
        
        if cur_r <= -1000:
            print("<-----!!!----->")
        
        print('%d:r:%f,p:%s,n:%s,tr:%s,ps:%s, rev:%s,dbuf:%s,ch:%s,ibuf:%s' % (j, cur_r, cur_p, cur_n, cur_ts, cur_ps, cur_rs, cur_ds, cur_ch, cur_init_ds_ep))
        
    print('r:%.4f,p:%.4f,tr:%.4f,pr:%.4f,rev:%.4f,dbuf:%.4f,ch:%.8f,ibuf:%d' % (cur_r_ep/MAX_EPISODE_LEN, cur_p_ep/MAX_EPISODE_LEN, cur_ts_ep/MAX_EPISODE_LEN, cur_ps_ep/MAX_EPISODE_LEN, cur_rs_ep/MAX_EPISODE_LEN, cur_ds_ep/MAX_EPISODE_LEN, cur_ch_ep/MAX_EPISODE_LEN, cur_init_ds_ep[0]))
        
def plot_everything(res, win=10):
    length = len(res)
    temp = np.array(res)
    
    rewards = temp[:,:,0]
    avg_r = np.sum(rewards, axis=1)/rewards.shape[1]
    plt.plot(range(avg_r.shape[0]), avg_r)
    
    avg_r_sm = moving_average(avg_r, win)
    plt.plot(range(avg_r_sm.shape[0]), avg_r_sm)
    
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
    
    powers = temp[:,:,2]
    avg_p = np.sum(powers, axis=1)/powers.shape[1]
    plt.plot(range(avg_p.shape[0]), avg_p)
    
    avg_p_sm = moving_average(avg_p, win)
    plt.plot(range(avg_p_sm.shape[0]), avg_p_sm)
    
    plt.xlabel('step')
    plt.ylabel('power')
    plt.show()
    
    bufs = temp[:,:,7]
    avg_b = np.sum(bufs, axis=1)/bufs.shape[1]
    plt.plot(range(avg_b.shape[0]), avg_b)
    
    avg_b_sm = moving_average(avg_b, win)
    plt.plot(range(avg_b_sm.shape[0]), avg_b_sm)
    
    plt.xlabel('step')
    plt.ylabel('buffer length')
    plt.show()
    
    ofs = temp[:,:,9]
    avg_o = np.sum(ofs, axis=1)/ofs.shape[1]
    plt.plot(range(avg_o.shape[0]), avg_o)
    
    avg_o_sm = moving_average(avg_o, win)
    plt.plot(range(avg_o_sm.shape[0]), avg_o_sm)
    
    plt.xlabel('step')
    plt.ylabel('buffer length')
    plt.show()
    
    return avg_r, avg_p, avg_b, avg_o

def read_log(dir_path, user_idx=0):
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []
    avg_ps = []
    avg_bs = []
    avg_os = []

    for name in fileList:
        path = dir_path + name
        res = np.load(path)

        temp_rs = np.array(res['arr_0'])
        avg_rs.append(temp_rs[:, user_idx])
        
        temp_ps = np.array(res['arr_1'])
        avg_ps.append(temp_ps[:, user_idx])
        
        temp_bs = np.array(res['arr_2'])
        avg_bs.append(temp_bs[:, user_idx])
        
        temp_os = np.array(res['arr_3'])
        avg_os.append(temp_os[:, user_idx])
    
    avg_rs = np.array(avg_rs)
    avg_ps = np.array(avg_ps)
    avg_bs = np.array(avg_bs)
    avg_os = np.array(avg_os)
    
    return avg_rs, avg_ps, avg_bs, avg_os
    
def plot_curve(rs, ps, bs, os, win=10):
    for avg_r in rs:
        avg_r_sm = moving_average(avg_r, win)
        plt.plot(range(avg_r.shape[0]), avg_r)
        plt.plot(range(avg_r_sm.shape[0]), avg_r_sm)
        plt.xlabel('step')
        plt.ylabel('Total moving reward')
    plt.show()
    
    for avg_p in ps:
        avg_p_sm = moving_average(avg_p, win)
        plt.plot(range(avg_p.shape[0]), avg_p)
        plt.plot(range(avg_p_sm.shape[0]), avg_p_sm)
        plt.xlabel('step')
        plt.ylabel('power')
    plt.show()
    
    for avg_b in bs:
        avg_b_sm = moving_average(avg_b, win) 
        plt.plot(range(avg_b.shape[0]), avg_b)
        plt.plot(range(avg_b_sm.shape[0]), avg_b_sm)
        plt.xlabel('step')
        plt.ylabel('buffer length')
    plt.show()
    
    for avg_o in os:
        avg_o_sm = moving_average(avg_o, win) 
        plt.plot(range(avg_o.shape[0]), avg_o)
        plt.plot(range(avg_o_sm.shape[0]), avg_o_sm)
        plt.xlabel('step')
        plt.ylabel('overflow probability')
    plt.show()
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#     import matplotlib.pyplot as plt
#     N = 8
#     y = np.zeros(N)
#     x1 = np.linspace(0, 10, N, endpoint=True)
#     x2 = np.linspace(0, 10, N, endpoint=False)
#     plt.plot(x1, y, 'o')

#     plt.plot(x2, y + 0.5, 'o')

#     plt.ylim([-0.5, 1])

#     plt.savefig('ex.eps', format='eps', dpi=1000)
#     plt.show()
            