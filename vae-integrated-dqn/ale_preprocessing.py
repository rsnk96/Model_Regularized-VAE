import numpy as np
#import pillow as Image


def preproc_unaltered(imgs):
    return [np.concatenate(phi,axis=-1) for phi in imgs]

def pong_remove_score(obs):
    obs[:30,:,:]=0
    return obs

def preproc_pong_noscore(imgs) :
    return [pong_remove_score(np.concatenate(phi, axis=-1)) for phi in imgs]