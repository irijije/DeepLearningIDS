import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from meta_data import *


os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def save_feature_dist(data, data2):
    FEATURE_LIST.remove('Label')
    for feature in FEATURE_LIST:
        fig, dist = plt.subplots()
        plt.title(feature)
        dist.hist([100*data[feature], 100*data2[feature]], bins=range(0, 100), label=['dos', 'benign'])
        dist.set_xlabel('normalized value')
        dist.set_ylabel('# of traffics')
        dist.legend(loc='upper right')
        plt.savefig(BASE_PATH+'feature_dist/'+feature.replace('/', '-'))
        plt.close()

def norm(data, stats):
    #return (data - stats['mean']) / stats['std']
    return (data-stats['min']) / (stats['max']-stats['min']+0.00001)

def load_dataset(filepath):
    df = pd.read_csv(filepath, sep=',', names=DOS_FEATURE_LIST, header=0, dtype=np.float32)
    #df = pd.read_csv(filepath)
    df = df.dropna()
    labels = df.pop('Label')
    df = df.astype('float32')
    #df.describe().to_csv("norm_params.csv")
    df = norm(df, df.describe().transpose())

    return df, labels

def show_result(hist):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], 'r', label='train loss')
    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    loss_ax.plot(hist.history['val_loss'], 'y', label='val loss')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    plt.show()

def train_model():
    train_data, train_labels = load_dataset(BASE_PATH+"CICIDS2018_small_dos_train.csv")
    #val_dataset = load_dataset(BASE_PATH+"CICIDS2018_val_dos.csv")
    #test_dataset = load_dataset(BASE_PATH+"CICIDS2018_test_dos.csv")
    
    #save_feature_dist(train_data, train_data2)

    METRICS = [
        #keras.metrics.TruePositives(name='tp'),
        #keras.metrics.FalsePositives(name='fp'),
        #keras.metrics.TrueNegatives(name='tn'),
        #keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        #keras.metrics.AUC(name='auc'),
    ]

    model = tf.keras.models.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(25,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(
        #optimizer='adam',
        optimizer=sgd,
        loss='binary_crossentropy',
        #metrics=['accuracy'],
        metrics=METRICS,
    )
    
    print(model.summary())

    hist = model.fit(train_data, train_labels,
                batch_size=200,
                validation_split=0.2,
                epochs=50,
                #class_weight = {0: 0.5, 1: 0.5},
    )

    
    show_result(hist)
    
    model.save("models/IDS_small_dos_v1.h5")

    #model.evaluate(test_dataset, verbose=2)

def test_model():
    test_data, test_labels = load_dataset(BASE_PATH+"CICIDS2018_small_dos_test.csv")
    model = keras.models.load_model("models/IDS_small_dos_v1.h5")
    loss, acc, precision, recall = model.evaluate(test_data,  test_labels, verbose=2)
    print("{:5.2f}%".format(100*acc))


if __name__ == "__main__":
    train_model()
    test_model()