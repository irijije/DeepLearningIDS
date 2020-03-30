import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_PATH = "CICIDS_dataset/"
FILE_PATH = "Processed Traffic Data for ML Algorithms/"


def make_dataset():
    # all_data = []
    # for f in glob.glob(BASE_PATH+FILE_PATH+"*.csv"):
    #     data = pd.read_csv(f, index_col=None)
    #     all_data.append(data)
    # data_concat = pd.concat(all_data, axis=0, ignore_index=True)
    # data_concat.to_csv(BASE_PATH+"CICIDS2018_all.csv", index=False)
    data = pd.read_csv(BASE_PATH+"CICIDS2018_all.csv")
    x = data.drop(['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp'], axis=1)
    x_train, x_test = train_test_split(x, test_size=0.2)
    pd.DataFrame(x_train).to_csv(BASE_PATH+"CICIDS2018_train.csv", index=None)
    pd.DataFrame(x_test).to_csv(BASE_PATH+"CICIDS2018_test.csv", index=None)


if __name__ == "__main__":
    make_dataset()