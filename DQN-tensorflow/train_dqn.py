import tensorflow as tf
import numpy as np
import numpy.random as rn
from misc.dqn import DQN
from misc.utils import toNatureDQNFormat
from misc.environment import environment
from misc.replay_memory import memory_element
from misc.replay_memory import replay_memory

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--replay_memory', default=1000000, type=int)
parser.add_argument('--total_frames', default=50000000, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--game', default='Breakout-v0')
parser.add_argument('--init_epsilon', default=1, type=float)
parser.add_argument('--final_epsilon', default=0.1, type=float)
parser.add_argument('--final_exploration_frame', default=1000000, type=int)
parser.add_argument('--start_training', default=50000, type=int)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--seed', default=74846, type=int)

cmd_params = parser.parse_args()
rn.seed(cmd_params['seed'])

epsilon = cmd_params['init_epsilon']
delta_epsilon = (cmd_params['final_epsilon']-epsilon)/(1.0*cmd_params['final_exploration_frame'])
gamma = cmd_params['gamma']

# Create the environment
env = environment({'game': cmd_params['game']})

# A list containing the most recent transitions
replay = replay_memory({'memory_size': cmd_params['memory_size'], \
                        'rn_seed': cmd_params['seed']})

params = {'nActions': env.nActions}
# DQN to be updated
dqn_s = DQN(params)
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

for iters in range(0, cmd_params['total_frames']):
    if iters <= cmd_params['start_training']:
       if rn.random() <  
