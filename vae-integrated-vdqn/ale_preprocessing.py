import numpy as np
#import pillow as Image
import Image

def resizephi(phi) :
    return [np.array(Image.fromarray(image).resize((84,84),Image.BILINEAR)) for image in phi]

def preproc_unaltered(imgs):
    return [np.concatenate(phi,axis=-1) for phi in imgs]

def pong_remove_score(obs):
    obs[:12,:,:]=0
    return obs

def preproc_pong_noscore(imgs) :
    return [pong_remove_score(np.concatenate(resizephi(phi), axis=-1))/np.float32(255) for phi in imgs]