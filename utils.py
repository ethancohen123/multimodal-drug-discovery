import torch
from torch import optim
import logging
import os
import json
import numpy as np


                
def save_img_as_npz(img, fname):
    img = img.numpy()
    img = img.transpose(1,2,0)
    img = img + 0.5 # undo normalization
    img_dict = {}
    img_dict['sample'] = img

    np.savez(fname, **img_dict)



def create_dir(dirname):
    try:
        os.mkdir(dirname)
        return True
    except OSError:
        return False