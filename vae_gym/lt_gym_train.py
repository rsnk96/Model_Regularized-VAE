import gym
import pickle
import Image
import numpy as np
import rlsvi
import tensorflow as tf

# Extract Y channel from RGB, resize to 84x84 and return
def toNatureDQNFormat(frame) :
	return np.array(Image.fromarray(frame).convert('YCbCr').resize((84,84),Image.BILINEAR))[:,:,0]/np.float32(255.0)

def RLVSI_preprocess(frame) :
	colframe = np.array(Image.fromarray(frame).resize((42,42),Image.BILINEAR))/np.float32(255.0)
	return np.concatenate([colframe.flatten(),[1]])

import gym
env = gym.make('IceHockey-v0')

num_episodes = 500
num_steps = 15000


na = env.action_space.n
ns = 42*42*3+1

print na,ns

sigma=1.0
lmb=1.0
gamma=0.99

R = rlsvi.rlsvi(ns,na,sigma,lmb)

print 'here'

sess = tf.InteractiveSession()
try:
    sess.run(tf.global_variables_initializer())
except AttributeError:
    sess.run(tf.initialize_all_variables())

for episode in range(num_episodes):
	obs = env.reset()
	obs_pp = RLVSI_preprocess(obs)
	for time_step in range(num_steps):
		if time_step % 1000 == 0 :
			print 'Episode %d Time step %d'%(episode+1,time_step+1)
		env.render()
		action = R.choose_action(obs_pp)
		next_obs, reward, done, info = env.step(action)
		next_obs_pp = RLVSI_preprocess(next_obs)
		R.add_data(obs_pp,action,reward,next_obs_pp,done)
		obs_pp=next_obs_pp
		if reward!=0 :
			print reward
		if done:
			break
	R.new_episode(sess)
