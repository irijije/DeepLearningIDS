import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras

from meta_data import *


os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def test_attack():
    x = pd.read_csv("dos.csv")
    x = x[DOS_FEATURE_LIST]
    x = x.dropna()
    y = x.pop('Label')
    x = x.astype('float32')

    df = pd.read_csv(BASE_PATH+"CICIDS2018_small_dos.csv")
    df = df[DOS_FEATURE_LIST]
    df = df.dropna()
    df.pop('Label')
    df = df.astype('float32')
    norm_params = df.describe().transpose()

    x = (x-norm_params['min']) / (norm_params['max']-norm_params['min']+0.00001)    
    x = tf.convert_to_tensor(x.iloc[0])[None, ...]
    model = keras.models.load_model("models/IDS_small_dos_alpha.h5")

    loss_object = tf.keras.losses.BinaryCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x)
        loss = loss_object(x, pred)
    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)

    for i in range(10):
        print("\n5 features changed. test: "+str(i+1))
        chosed_features = np.zeros(25)
        nums = np.random.choice(25, 5, replace=False)
        for num in nums:
            chosed_features[num] = 1
        epsilons = [0, 0.01, 0.05, 0.1, 0.5, 1]
        for eps in epsilons:
            adv_x = x + eps*signed_grad[0]*chosed_features.transpose()
            print("{0:3.2f} changed. predicion: {1:5.4f}".format(eps, model.predict(adv_x)[0][0]))
            print("x: ")
            print(x)
            print("adv_x: ")
            print(adv_x)



if __name__ == "__main__":
    test_attack()