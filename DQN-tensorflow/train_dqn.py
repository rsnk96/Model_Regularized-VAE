import tensorflow as tf
import numpy as np
import numpy.random as rn
import sys
from misc.dqn import DQN
from misc.utils import toNatureDQNFormat
from misc.environment import environment
from misc.replay_memory import memory_element
from misc.replay_memory import replay_memory

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--memory_size', default=1000000, type=int)
parser.add_argument('--total_frames', default=50000000, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--game', default='Breakout-v0')
parser.add_argument('--init_epsilon', default=1, type=float)
parser.add_argument('--final_epsilon', default=0.1, type=float)
parser.add_argument('--final_exploration_frame', default=1000000, type=int)
parser.add_argument('--start_training', default=50000, type=int)
parser.add_argument('--target_update_every', default=10000, type=int)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--seed', default=74846, type=int)
parser.add_argument('--alpha', default=0.95, type=float, help='momentum for rmsprop')
parser.add_argument('--learning_rate', default=0.00025, type=float, help='learning rate for rmsprop')
parser.add_argument('--eps', default=0.01, type=float, help='minimum denominator value for rmsprop')
parser.add_argument('--image_format', default='NHWC', help='[NHWC | NCHW]')
parser.add_argument('--history_length', default=4, type=int)

cmd_params = vars(parser.parse_args())
rn.seed(cmd_params['seed'])

epsilon = cmd_params['init_epsilon']
delta_epsilon = (cmd_params['final_epsilon']-epsilon)/(1.0*cmd_params['final_exploration_frame'])
gamma = cmd_params['gamma']
shape = [84, 84, 4]

# Create the environment
env = environment({'game': cmd_params['game'],\
                   'image_format': cmd_params['image_format'],\
                   'history_length': cmd_params['history_length']})

# A list containing the most recent transitions
replay = replay_memory({'memory_size': cmd_params['memory_size'], \
                        'rn_seed': cmd_params['seed'], 'shape': shape})

params = {'nActions': env.nActions}
# DQN to be updated
dqn_s = DQN(params)
dqn_s.createOptimizer({'learning_rate': cmd_params['learning_rate'], \
        'alpha': cmd_params['alpha'], 'epsilon': cmd_params['eps']})

# target DQN
dqn_t = DQN(params)

# Function to copy weights from source dqn to target dqn
assign_ops = {}
for k in dqn_s.weights.keys():
    assign_ops[k] = dqn_t.weights[k].assign(dqn_s.weights[k])
    
def copy_network(sess):
    for k in assign_ops:
        sess.run(assign_ops[k])

sess = tf.InteractiveSession()

try:
    sess.run(tf.global_variables_initializer())
except AttributeError:
    sess.run(tf.initialize_all_variables())

# Copy weights to target network
copy_network(sess)

for iters in range(1, cmd_params['total_frames']+1):
    # state before taking action s_{t}
    if env.done:
        env.reset_env()
    
    s = env.state_history.history
    if rn.random() < epsilon:
        action_taken = env.take_random_step()
    else:
        action_taken = dqn_s.getAction(sess, np.reshape(s, [1]+shape))[0]
    env.take_step(action_taken)
    replay.add_transition(s, action_taken, env.reward, env.state_history.history, env.done)
    

    if iters >= cmd_params['start_training']:
        s, a, r, sp, t = replay.get_transitions(cmd_params['batch_size']) 
        qsp = dqn_t.getMaxActionValue(sess, sp)
        target = gamma*(1-t)*qsp + r
        _, loss = sess.run([dqn_s.optimize, dqn_s.loss] , {dqn_s.phi: s, \
                            dqn_s.target: target, dqn_s.input_actions: a})
        if iters%1000 == 0:
            print('Iteration #: %.8d , Loss: %.6f'%(iters, loss))
    elif iters%1000 == 0:
        print('Iteration #: %.8d'%(iters))

    if iters <= cmd_params['final_exploration_frame']:
        epsilon = epsilon + delta_epsilon

    if iters%cmd_params['target_update_every'] == 0:
        copy_network(sess)
