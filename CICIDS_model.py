import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
BASE_PATH = "CICIDS2018_dataset/"
MODEL_NAME = "models/IDS_small_v1"
le = LabelEncoder()


def norm(train_dataset, train_stats):
    return (train_dataset - train_stats['mean']) / train_stats['std']

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    #df['Label'] = pd.Categorical(df['Label'])
    #df['Label'] = df.Label.cat.codes
    target = df.pop('Label')
    le.fit(target)
    target = le.transform(target)
    df = df.astype('float32')
    #dataset = norm(df, df.describe().transpose())
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target))
    dataset = dataset.shuffle(len(df)).batch(50)

    return dataset

def show_result(hist):
    fig, loss_ax = plt.subplots()
    fig = fig
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], 'r', label='train loss')
    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    #loss_ax.plot(hist.history['val_loss'], 'y', label='val loss')
    #acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    plt.show()

def train_model():
    train_dataset = load_dataset(BASE_PATH+"CICIDS2018_test_small.csv")
    #test_dataset = load_dataset(BASE_PATH+"CICIDS2018_test.csv")

    # METRICS = [
    #     keras.metrics.TruePositives(name='tp'),
    #     keras.metrics.FalsePositives(name='fp'),
    #     keras.metrics.TrueNegatives(name='tn'),
    #     keras.metrics.FalseNegatives(name='fn'), 
    #     keras.metrics.BinaryAccuracy(name='accuracy'),
    #     keras.metrics.Precision(name='precision'),
    #     keras.metrics.Recall(name='recall'),
    #     keras.metrics.AUC(name='auc'),
    # ]

    model = tf.keras.models.Sequential([
        keras.layers.Dense(64, activation='relu'),
        #keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu'),
        #keras.layers.Dropout(0.5),
        keras.layers.Dense(15, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        #metrics=METRICS,
    )

    hist = model.fit(train_dataset,
                #validation_split = 0.2,
                epochs=5,
            )

    show_result(hist)
    
    model.save(MODEL_NAME)

    #model.evaluate(test_dataset, verbose=2)


if __name__ == "__main__":
    train_model()