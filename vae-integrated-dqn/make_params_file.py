import tensorflow as tf
import pdb
import numpy as np
import numpy.random as rn
import os
import scipy.misc
import argparse
import pickle



params = {}
params['batch_size'] = 20
params['X_size'] = [210,160,12]
params['z_size'] = 5
params['beta'] = 1
params['learn_rate'] = 1e-4


aux_data = {'params': params}
dump_path = 'params/pong'
dump_dir = '%s_%f_%d'%(dump_path, params['beta'], params['z_size'])
os.system('mkdir -p %s'%(dump_dir))
pickle.dump(aux_data, open('%s/aux_data.pkl'%(dump_dir), 'w'))