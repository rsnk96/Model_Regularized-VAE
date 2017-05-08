import numpy as np
import Image

def preproc_unaltered(imgs) :
    return [np.concatenate(phi,axis=-1) for phi in imgs]
