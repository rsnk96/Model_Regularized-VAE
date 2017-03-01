# Class to store past k frame history to feed as input to the DQN
import numpy as np

class history(object):

	def __init__(self, params):
		self.history_length = params['history_length']
		self.start_frame = params['start_frame'] # Must be HxW array
		self.W = self.start_frame.shape[1]
		self.H = self.start_frame.shape[0]
		# NHWC or NCWH
		self.image_format = params.get('image_format', 'NHWC')
		if self.image_format == 'NHWC':
			self.history = np.repeat(np.reshape(self.start_frame, [self.H, self.W, 1]), self.history_length, 2)
		elif self.image_format == 'NCWH':
			self.history = np.repeat(np.reshape(self.start_frame, [1, self.H, self.W]), self.history_length, 0)
		else:
			raise "Invalid image format! Must be NHWC or NCWH"

	def update(frame):
		if self.image_format == 'NHWC':
			self.history[:,:,0:3] = self.history[:,:,1:4].copy()
			self.history[:,:,3] = frame.copy()
		else:
			self.history[0:3,:,:] = self.history[1:4,:,:].copy()
			self.history[3,:,:] = frame.copy()
	
	def reset(frame):
		self.start_frame = frame
		if self.image_format == 'NHWC':
			self.history = np.repeat(np.reshape(self.start_frame, [self.H, self.W, 1]), self.history_length, 2)
		elif self.image_format == 'NCWH':
			self.history = np.repeat(np.reshape(self.start_frame, [1, self.H, self.W]), self.history_length, 0)


