import tensorflow as tf
import numpy as np
import numpy.random as rn

# Choosing appropriate matrix multiplication function
if tf.__version__ == '0.10.0':
	mmul=tf.mul
else :
	mmul=tf.matmul

mu = tf.placeholder(tf.float32,[None])
covmat = tf.placeholder(tf.float32,[None,None])
eps = tf.placeholder(tf.float32,[None])
mvn_transform_op = tf.reshape(mmul(covmat,tf.reshape(eps,[-1,1])),[-1])+mu

def mvn_draw(sess,m,s) :
	return sess.run(mvn_transform_op,feed_dict={mu:m,covmat:s,eps:rn.normal(0,1,m.size)})

class bayesian_regressor(object) :
	# Fit a bayesian least squares regression model with a Gaussian prior
	# and get the estimated coefficients and the covariance
	def __init__(o,k,sigma=1.0,lmb=1.0) :
		o.X=[]
		o.y=[]
		o.k=k
		o.sigma=sigma
		o.sigmasq=sigma*sigma
		o.lmb=lmb
		o.theta_est=np.zeros([o.k], dtype=np.float32)
		o.cov_est=o.sigmasq*np.eye(o.k, dtype=np.float32)
		o.theta_sample=np.dot(o.cov_est,rn.normal(0,1,k))+o.theta_est
		# Tensorflow ops for computing the fit and covariance in estimate
		o.A = tf.placeholder(tf.float32,[None,o.k])
		o.b = tf.placeholder(tf.float32,[None])
		o.cov = tf.matrix_inverse(mmul(tf.transpose(o.A),o.A)/o.sigmasq+o.lmb*tf.eye(o.k,o.k))
		o.theta = mmul(o.cov,mmul(tf.transpose(o.A),tf.reshape(o.b,[-1,1]))/o.sigmasq)
	# Store given data to be fit later
	def add_data(o,x,yi):
		o.X.append(x)
		o.y.append(yi)
	# Reset stored data to start a new round of fitting
	def reset_data(o):
		o.X=[]
		o.y=[]
	# Perform fit
	def fit(o,sess) :
		if len(o.X)==0:
			return
		o.theta_est,o.cov_est = sess.run([o.theta,o.cov],feed_dict={o.A:o.X,o.b:o.y})
		o.theta_est = o.theta_est.flatten()
	# Sample and store a set of parameters from the learnt Gaussian posterior
	def sample_from_posterior(o,sess):
		o.theta_sample=mvn_draw(sess,o.theta_est,o.cov_est)
	# Get regression output for given input
	def evaluate(o,x) :
		return np.dot(o.theta_sample,x)

class rlsvi(object):
	# Stationary RLSVI algorithm implementation
	def __init__(o,ns,na,sigma=1.0,lmb=1.0,gamma=0.99) :
		o.sigma = sigma
		o.sigmasq = sigma*sigma
		o.lmb = lmb
		o.gam = gamma
		o.ns = ns
		o.na = na
		# One regression model for each action
		o.models = [bayesian_regressor(ns,o.sigma,o.lmb) for i in range(o.na)]
	# Compute the estimate and reset the data stored in the models for a
	# new episode.
	def new_episode(o,sess):
		for i in range(o.na) :
			o.models[i].fit(sess)
			o.models[i].sample_from_posterior(sess)
			o.models[i].reset_data()
	# Return the best action according to current estimates for given state
	def choose_action(o,s):
		return np.argmax(([o.models[i].evaluate(s) for i in range(o.na)]))
	# Save state transition and reward data
	def add_data(o,s1,a,r,s2,done) :
		# Compute the targets
		if done :
			bi=r
		else :
			bi=r+o.gam*np.max(([o.models[i].evaluate(s2) for i in range(o.na)]))
		# Store in model for the given action
		o.models[a].add_data(s1,bi)
