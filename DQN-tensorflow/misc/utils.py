import Image
import numpy as np

def toNatureDQNFormat(frame):
    return np.array(Image.fromarray(frame).convert('YCbCr').resize((84,84), Image.BILINEAR))[:,:,0]/np.float32(255.0)
