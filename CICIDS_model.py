import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


MODEL_NAME = "IDSv1"
BASE_PATH = "CICIDS_dataset/"
FILE = "all.csv"


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=50,
        label_name='Label',
        na_value="?",
        num_epochs=200,
        ignore_errors=True,
        **kwargs)

    return dataset

def load_dataset():
    print("\nloading dataset...\n")
    
    LABELS = [0, 1]
    raw_train_data = get_dataset(BASE_PATH+"CICIDS2018_train.csv")
    raw_test_data = get_dataset(BASE_PATH+"CICIDS2018_test.csv")

    print("\nsuccess\n")

def train_model(dataset):
    train_traffics, test_traffics = dataset[0], dataset[1]
    train_labels, test_labels = dataset[1], dataset[2]

    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(10,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='spares_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_traffics, train_labels, epochs=100,
        validation_data = (test_traffics, test_labels),
    )

    loss, acc = model.evaluate(test_traffics, test_labels, verbose=2)
    print(loss, acc)

    model.save(MODEL_NAME)


if __name__ == "__main__":
    pass