import tensorflow as tf
import numpy as np
import numpy.random as rn
import numpy.linalg as la

# Choosing appropriate matrix multiplication function
if tf.__version__ == '0.10.0':
	mmul=tf.mul
else :
	mmul=tf.matmul

mu = tf.placeholder(tf.float32,[None])
covmat = tf.placeholder(tf.float32,[None,None])
eps = tf.placeholder(tf.float32,[None])
mvn_transform_op = tf.reshape(mmul(covmat,tf.reshape(eps,[-1,1])),[-1])+mu

A_p = tf.placeholder(tf.float32,[None,None])
B_p = tf.placeholder(tf.float32,[None,None])
tfdot_op = mmul(A_p,B_p)
tfadd_op = A_p + B_p

def mvn_draw(sess,m,s) :
	return sess.run(mvn_transform_op,feed_dict={mu:m,covmat:s,eps:rn.normal(0,1,m.size)})

def tfdot(sess,A,B) :
	return sess.run(tfdot_op,feed_dict = {A_p:A, B_p:B})

def tfadd(sess,A,B) :
	return sess.run(tfadd_op,feed_dict = {A_p:A, B_p:B})

class bayesian_regressor(object) :
	# Fit a bayesian least squares regression model with a Gaussian prior
	# and get the estimated coefficients and the covariance
	def __init__(o,k,sigma=1.0,lmb=1.0,capacity=20000) :
		o.k = k
		o.phi = []
		o.phi_nxt = []
		o.phi_ep = [] # Data for current episode to be used for incremental updates
		o.r = []
		o.r_ep = []
		o.done = []
		o.done_ep = []
		o.mc=capacity
		o.sigma=sigma
		o.sigmasq=sigma*sigma
		o.lmb=lmb
		o.theta_est=np.zeros([o.k], dtype=np.float32)
		o.cov_est=np.eye(o.k, dtype=np.float32)/o.lmb
		o.theta_sample=np.dot(o.cov_est,rn.normal(0,1,k))+o.theta_est
		# The inverse of the covariance matrix is updated after each episode
		o.covmat_inv=np.eye(o.k, dtype=np.float32)*o.lmb
	def reset_data(o) :
		# Clear the buffer for the current episode
		o.phi_ep=[]
	# Store given data to be fit later
	def add_data(o,phi_t,r_t,phi_t1,done):
		o.phi_ep.append(phi_t)
		o.phi.append(phi_t)
		o.phi_nxt.append(phi_t1)
		o.r.append(r_t)
		# If the episode has ended, add the entry for the next state alsp
		if done :
			o.done.append(0)
			o.done.append(1)
			o.phi_ep.append(phi_t1)
			o.phi.append(phi_t1)
			o.phi_nxt.append(phi_t1)
			o.r.append(0)
		else :
			o.done.append(0)
	# Perform fit
	def fit(o,sess,q_nxt) :
		if len(o.phi_ep)!=0 :
			phi_new_mat = np.reshape(o.phi_ep,(-1,o.k))
			phitphi = tfdot(sess,phi_new_mat.T,phi_new_mat)
			# Update the inverse of the covariance matrix
			o.covmat_inv+=phitphi/o.sigmasq
			# Find the new covariance of the estimate
			o.cov_est = la.inv(o.covmat_inv)
			# Compute targets from the next state-action values given
			trg = np.reshape(q_nxt + o.r,(-1,1))
			phi_mat = np.reshape(o.phi,(-1,o.k))
			theta_est_new = np.dot(o.cov_est,tfdot(sess,phi_mat.T,trg)).flatten()/o.sigmasq
			#print 'norm',la.norm(theta_est_new - o.theta_est)/la.norm(o.theta_est)
			o.theta_est = theta_est_new
			#np.set_printoptions(3)
			#print np.hstack([o.phi,np.array(o.r).reshape((-1,1)),np.array(o.done).reshape((-1,1))])
			#np.set_printoptions()
			#print '==============='
		#print o.theta_est
	# Sample and store a set of parameters from the learnt Gaussian posterior
	def sample_from_posterior(o,sess):
		#o.theta_sample=mvn_draw(sess,o.theta_est,o.cov_est)
		#np.dot(o.cov_est,rn.normal(0,1,o.k))+o.theta_est
		o.theta_sample = rn.multivariate_normal(o.theta_est,o.cov_est)
	# Get regression output for given input
	def evaluate(o,x) :
		return np.dot(x,o.theta_sample)

class rlsvi(object):
	# Stationary RLSVI algorithm implementation
	def __init__(o,ns,na,sigma=1.0,lmb=1.0,gamma=0.99) :
		o.sigma = sigma
		o.sigmasq = sigma*sigma
		o.lmb = lmb
		o.gam = gamma
		o.ns = ns
		o.na = na
		o.tau = 1024.0
		o.act_counter = np.zeros(o.na)
		# One regression model for each action
		o.models = [bayesian_regressor(ns,o.sigma,o.lmb) for i in range(o.na)]
	# Compute the estimate and reset the data stored in the models for a
	# new episode.
	def new_episode(o,sess):
		theta_all = np.vstack([o.models[i].theta_sample for i in range(o.na)]).T
		q_nxt_list = []
		# Compute the next state action values which will be used to
		# find the targets for the linear regression
		for a in range(o.na) :
			if len(o.models[a].phi_nxt) :
				qsa = tfdot(sess,np.reshape(o.models[a].phi_nxt,(-1,o.ns)),theta_all)
				q_nxt = o.gam*np.max(qsa,axis=1)
				q_nxt[np.where(np.array(o.models[a].done))]=0
			else :
				q_nxt = None
			q_nxt_list.append(q_nxt)
		for i in range(o.na) :
			#print 'action',i
			o.models[i].fit(sess,q_nxt_list[i])
			o.models[i].sample_from_posterior(sess)
			o.models[i].reset_data()
		#for i in range(o.na) :
		#	print 'Action %d'%(i),o.models[i].theta_sample
		if o.tau > 0.01 :
			o.tau/=4
		#print 'Action %', o.act_counter/np.sum(o.act_counter)
		#o.act_counter = np.zeros(o.na)
	# Return the best action according to current estimates for given state
	def choose_action(o,s):
		qsa = np.array([o.models[i].evaluate(s) for i in range(o.na)])/o.tau
		if 0 : #o.tau > 1 :
			qsa -= np.mean(qsa)
			probs = np.exp(qsa)
			probs /= np.sum(probs)
			action = int(rn.choice(o.na,p=probs))
		else :
			action = np.argmax(qsa)
		#o.act_counter[action]+=1
		#print 'action',action
		return action
	# Save state transition and reward data
	def add_data(o,s1,a,r,s2,done,sess=None) :
		# Store in model for the given action
		o.models[a].add_data(s1,r,s2,done)
	def print_params(o):
		for i in range(o.na) :
			print 'action',i
			np.set_printoptions(3)
			print o.models[i].theta_est
			print '=====Covariance of Estimate====='
			print o.models[i].cov_est
			print '================================'
			np.set_printoptions()
