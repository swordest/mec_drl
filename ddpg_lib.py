import tensorflow as tf
import numpy as np
import tflearn
from collections import deque
import random
import ipdb as pdb

# setting for hidden layers 
Layer1 = 400
Layer2 = 300

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, user_id=''):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.user_id = user_id

        start_idx = len(tf.global_variables())
        
        start_idx_train = len(tf.trainable_variables())
        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()[start_idx_train:]
        
        end_idx_train = len(tf.trainable_variables())

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[end_idx_train:]
        
        self.params = tf.global_variables()[start_idx:]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

#         # Combine the gradients here
#         self.unnormalized_actor_gradients = tf.gradients(
#             self.scaled_out, self.network_params, -self.action_gradient)
#         self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        
        # Combine the gradients here
        grads = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        # the gradient shoudl not be None
        self.unnormalized_actor_gradients = [grad if grad is not None else tf.zeros_like(var) 
                                             for var, grad in zip(self.network_params, grads)]
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim], name="input_"+str(self.user_id))
        net = tflearn.fully_connected(inputs, Layer1)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, Layer2)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='sigmoid', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound, name="output_"+str(self.user_id))
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
        
    def init_target_network(self, path):
        res = np.load(path)
        weights_load = np.array(res['arr_0'])

        for i in range(len(weights_load)):
            self.sess.run(self.params[i].assign(tf.constant(weights_load[i])))

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class ActorNetworkLD(object):
    """
    Actor network loaded from stored models
    """
    
    def __init__(self, sess, user_id):
        self.sess = sess
        self.user_id = user_id
        
        graph = tf.get_default_graph()
        self.inputs = graph.get_tensor_by_name("input_"+user_id+"/X:0")
        self.scaled_out = graph.get_tensor_by_name("output_"+user_id+":0")
        
    def predict(self, s):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: s
        })

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, Layer1)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, Layer2)
        t2 = tflearn.fully_connected(action, Layer2)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
        

class DeepQNetwork(object):
    """
    Input to the network is the state, output is a vector of Q(s,a).
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, user_id):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.user_id = user_id

        # Create the critic network
        self.inputs, self.q_out = self.create_deep_q_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_q_out = self.create_deep_q_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.target_Q = tf.placeholder(tf.float32, [None, self.a_dim])
        
        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.target_Q, self.q_out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

    def create_deep_q_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim], name="input_"+str(self.user_id))
        net = tflearn.fully_connected(inputs, Layer1)
#         net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.elu(net)

        net = tflearn.fully_connected(net, Layer2)
#         net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.elu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
#         w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        q_out = tflearn.fully_connected(net, self.a_dim, name="output_"+str(self.user_id))
        return inputs, q_out

    def train(self, inputs, target_Q):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.target_Q: target_Q
        })

    def predict(self, inputs):
        q_out = self.sess.run(self.q_out, feed_dict={
            self.inputs: inputs
        })
        return np.argmax(q_out, axis=1), q_out

    def predict_target(self, inputs):
        q_out = self.sess.run(self.target_q_out, feed_dict={
            self.target_inputs: inputs
        })
        return np.argmax(q_out, axis=1), q_out

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.12, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
    
class GaussianNoise:
    def __init__(self, sigma0=1.0, sigma1=0.0, size=[1]):
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.size = size
        
    def __call__(self):
        self.sigma0 *= 0.9995
        self.sigma0 = np.fmax(self.sigma0, self.sigma1)
        return np.random.normal(0, self.sigma0, self.size)


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0



