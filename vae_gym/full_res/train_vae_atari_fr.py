from __future__ import print_function
import cv2
import tensorflow as tf
import pdb
import numpy as np
import numpy.random as rn
import os
import scipy.misc
import argparse
import pickle
import betavae_color as vae
from PIL import Image

# Magic seed number 
magic_seed_number = 74846

def sample_batch(frameslist, perm, n=1) :
    return [np.array(frameslist[perm[x]], dtype=np.float)/np.float32(255) for x in rn.choice(len(perm),n)]
    

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', default='data/breakout_frame_saver_test.pkl')
parser.add_argument('--tr_iters', default=20679, type=int)
parser.add_argument('--beta', default=1.28, type=float)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--dump_path', default='output/models')
parser.add_argument('--batch_size', default=20, type=int)

commandline_params = vars(parser.parse_args())

frames=[]
with open(commandline_params['data_file'], 'r') as f:
    frames = pickle.load(f)
f.close()

n_total = len(frames)
n_valid = int(.2*n_total)
rn.seed(magic_seed_number) # Seed chosen by die rolls. Guaranteed to be random
perm = rn.choice(n_total,n_total)
perm_train = perm[:-n_valid]
perm_valid = perm[-n_valid:]

sess = tf.InteractiveSession()

tr_iters = commandline_params['tr_iters']
lr_rate = commandline_params['learning_rate']
dump_path = commandline_params['dump_path']

params = {}
params['batch_size'] = commandline_params['batch_size']
params['X_size'] = [210,160,3]
params['z_size'] = 30
params['beta'] = commandline_params['beta']

print('---Params used---')
print(params)

params_generated = params

VAE = vae.vae(params)
VAE._create_network_()

train_step = tf.train.AdamOptimizer(lr_rate).minimize(VAE.total_loss)

try:
    sess.run(tf.global_variables_initializer())
except AttributeError:
    sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

aux_data = {'params': params, 'commandline_params': commandline_params, \
            'perm_train': perm_train, 'perm_valid': perm_valid, \
            'magic_seed_number': magic_seed_number} 

os.system('mkdir -p %s'%(dump_path))
pickle.dump(aux_data, open('%s/aux_data.pkl'%(dump_path), 'w'))

for i in range(tr_iters):
    
    batch = sample_batch(frames, perm_train, params['batch_size'])
    inputReshaped = [cv2.resize(frame,(160,210)) for frame in batch]

    _, loss_val = sess.run([train_step, VAE.total_loss], \
                               feed_dict = {VAE.X_placeholder: inputReshaped})
    

    if i % 10 == 0:
        print('Iteration %.4d  Train Loss: %6.3f'%(i+1, loss_val))
    
    if (i) % 1000 == 0:
        generated = VAE.generateSample(sess, n_samples=params['batch_size'])
        os.system('mkdir -p %s/iter_%.6d'%(dump_path, i+1))
        for im in range(params['batch_size']):
            reshaped_image = generated[im]
            reshaped_image = reshaped_image.reshape((210, 160,3))
            save_path = saver.save(sess, '%s/iter_%.6d/checkpoint.ckpt'%(dump_path, i+1))
            scipy.misc.toimage(reshaped_image, cmin=0.0, cmax=1.0).save('%s/iter_%.6d/img%.3d.png'%(dump_path, i+1, im))
            # cv2.imwrite('%s/iter_%.6d/img%.3dOG.png'%(dump_path, i+1, im), inputReshaped[im])
        print('Saved model to %s'%(save_path))
os.system('mkdir -p %s/iter_%.6d'%(dump_path, i+1))        
save_path = saver.save(sess, '%s/iter_%.6d/checkpoint.ckpt'%(dump_path, i+1))
