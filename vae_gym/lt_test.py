import gym
import pickle
import Image
import numpy as np
import rlsvi as rlsvi
import tensorflow as tf
import betavae_cnn_84 as vae
import argparse


sess = tf.InteractiveSession()
try:
    sess.run(tf.global_variables_initializer())
except AttributeError:
    sess.run(tf.initialize_all_variables())


num_episodes = 10000
num_steps = 50000

env = gym.make('CartPole-v0')
obs = env.reset()

na = env.action_space.n
#na = 3
na_true = env.action_space.n
#ns = 4*params['z_size']+1
ns = obs.size

print na,ns

sigma=2
lmb=0.005
gamma=0.99


def concatphi(phi) :
	#return np.concatenate([phi,[1]])
	return phi


R = rlsvi.rlsvi(ns,na,sigma,lmb)
# Use the following to hard code a set of parameters that give good performance
#R.models[0].theta_sample=np.array([0.1,0,-1.0,-2.0,0])
#R.models[1].theta_sample=np.array([-0.1,0,1.0,2.0,0])

preproc_fn = concatphi
best_score=0
ep_interval,ep_interval_dec = 1005,110
ep_clock = ep_interval
r_epoch = 0
epochs=0
for episode in range(num_episodes):
	obs = env.reset()
	ep_r = 0
	for time_step in range(num_steps):
		#if (episode+1)%200==0 :
		#env.render()
		action = R.choose_action(concatphi(obs))
		next_obs, reward, done, info = env.step(action)
		ep_r+=reward
		r_epoch+=reward
		R.add_data(concatphi(obs),action,reward,concatphi(next_obs),done,sess)
		obs = next_obs
		if done:
			epochs+=1
			if (episode)%100==0 :
				print 'Episode %d : Avg Epoch reward : %f'%(episode+1,r_epoch*1.0/epochs)
				epochs=0
				r_epoch=0
				R.print_params()
			break
	R.new_episode(sess)
