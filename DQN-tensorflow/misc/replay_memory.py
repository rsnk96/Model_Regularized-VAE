# Class that emulates the replay memory from DQN
from misc.history import history
import numpy as np
import numpy.random as rn

class memory_element(object):
	
	def __init__(self, s, a, r, sp):
		self.s = s
		self.a = a
		self.r = r
		self.sp = sp
	
class replay_memory(object):

	def __init__(self, params):
		self.memory_size = params['memory_size']
		self.rn_seed = params['rn_seed']
		self.shape = params['shape']
		rn.seed(self.rn_seed)
		self.memory = []
		self.batch_size = []

	def add_transition(self, s, a, r, sp):
		if len(self.memory) == self.memory_size:
			self.memory.pop(0)
		self.memory.append(memory_element(s, a, r, sp))
	
	def get_transitions(batch_size):
		#TODO: Make this more efficient
		sample_indices = rn.randint(0, len(self.memory), batch_size).tolist()
		states_t = np.zeros((batch_size, self.shape[0], \
							 self.shape[1], self.shape[2]))
		states_t_1 = np.zeros((batch_size, self.shape[0], \
							 self.shape[1], self.shape[2]))
		action_t = np.zeros((batch_size))
		reward_t_1 = np.zeros((batch_size))

		for s in sample_indices:
			curr_memory = self.memory[sample_indices[s]]
			states_t[s, :, :, :] = curr_memory.s
			states_t_1[s, :, :, :] = curr_memory.sp
			action_t[s] = curr_memory.a
			reward_t_1[s] = curr_memory.r

		return states_t, action_t, reward_t_1, states_t_1
