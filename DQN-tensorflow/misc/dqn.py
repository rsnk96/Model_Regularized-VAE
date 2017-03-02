import tensorflow as tf
import numpy as np

def getWeight(shape):
    return tf.Variable(tf.random_uniform(shape, minval=-0.08, maxval=0.08, dtype=tf.float32))

def getBias(shape):
    # Initialize biases as 0
    return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32), dtype=tf.float32)

class DQN(object):
    def __init__(self, params):
        self.nActions = params['nActions']
        self.phi = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
        self.target = tf.placeholder(tf.float32, shape=[None])
        self.input_actions = tf.placeholder(tf.int32, shape=[None])
        self.weights = {}
        self.createLayers()
		
    def createLayers(self):
        
        # Convolution layer 1
        self.weights['conv1_w'] = getWeight([8, 8, 4, 32])
        self.weights['conv1_b'] = getWeight([32])
        conv1 = tf.nn.relu(tf.nn.conv2d(self.phi, self.weights['conv1_w'], \
                strides=[1, 4, 4, 1], padding = 'VALID') \
                 + self.weights['conv1_b'])
        self.conv1 = conv1

        # Convolution layer 2
        self.weights['conv2_w'] = getWeight([4, 4, 32, 64])
        self.weights['conv2_b'] = getWeight([64])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['conv2_w'], \
                strides=[1, 2, 2, 1], padding = 'VALID') \
                + self.weights['conv2_b'])
        self.conv2 = conv2

        # Convolution layer 3
        self.weights['conv3_w'] = getWeight([3, 3, 64, 64])
        self.weights['conv3_b'] = getWeight([64])
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, self.weights['conv3_w'], \
                strides=[1, 1, 1, 1], padding = 'VALID') \
                + self.weights['conv3_b'])
        self.conv3 = conv3

        # Reshape layer
        conv3_reshaped = tf.reshape(conv3, [-1, 3136])

        # Fully connected layer 1
        self.weights['fc1_w'] = getWeight([3136, 512])
        self.weights['fc1_b'] = getBias([512])
        fc1 = tf.nn.relu(tf.matmul(conv3_reshaped, self.weights['fc1_w']) \
        		+ self.weights['fc1_b'])

        # Fully connected layer 2
        self.weights['fc2_w'] = getWeight([512, self.nActions])
        self.weights['fc2_b'] = getBias([self.nActions])
        fc2 = tf.matmul(fc1, self.weights['fc2_w']) + self.weights['fc2_b']
        self.fc2 = fc2

        # Action selection
        self.action_selection = tf.argmax(fc2, axis=1)
		
        # Action value selection
        self.action_value_selection = tf.reduce_max(fc2, axis=1)

        # Loss computation given target, states, actions
        gather_index = tf.stack([tf.range(tf.shape(fc2)[0]),\
                                 self.input_actions], axis=1)
        self.loss = tf.reduce_mean(tf.square(tf.clip_by_value(self.target - tf.gather_nd(fc2, gather_index), -1, 1)))
	
    def getAction(self, sess, state):
        return sess.run(self.action_selection, {self.phi:state})

    def getMaxActionValue(self, sess, state):
    	return sess.run(self.action_value_selection, {self.phi:state})

    def createOptimizer(self, params):
        lr = params['learning_rate']
        mom = params['alpha']
        eps = params['epsilon']
        self.optimizer = tf.train.RMSPropOptimizer(lr, momentum=mom, epsilon=eps)
        self.optimize = self.optimizer.minimize(self.loss)

