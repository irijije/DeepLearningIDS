import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras

from meta_data import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def test_attack():
    x = pd.read_csv("dos.csv", sep=',', names=DOS_FEATURE_LIST, header=0, dtype=np.float32)
    y = x.pop('Label')
    x = x.astype('float32')
    norm_params = pd.read_csv("norm_params.csv", sep=',', index_col=0, header=0)
    norm_params = norm_params.astype('float32')
    x_norm = (x-norm_params.T['min']) / (norm_params.T['max']-norm_params.T['min']+0.00001)
    model = keras.models.load_model("models/IDS_small_dos_v1.h5")
    print(x)
    print(norm_params.T['max'])
    print(x_norm)
    preds = model.predict(x_norm)
    print(preds)

    

if __name__ == "__main__":
    test_attack()