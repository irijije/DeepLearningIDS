import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from meta_data import *


def drop_columns():
    chunks = pd.read_csv(BASE_PATH+FILE_PATH+"Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", chunksize=1000000)
    df = []
    for chunk in chunks:
        print("chunk processing")
        chunk = chunk.drop(['Flow ID', 'Src IP', 'Src Port', 'Dst IP'], axis=1)
        df.append(chunk)
    pd.concat(df).to_csv(BASE_PATH+FILE_PATH+"Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", index=False)

def make_dataset():
    all_data = []
    for f in glob.glob(BASE_PATH+FILE_PATH+"*.csv"):
        data = pd.read_csv(f, index_col=None, header=0)
        all_data.append(data)
    data = pd.concat(all_data, axis=0, ignore_index=True)
    data = data.drop(['Timestamp'], axis=1)
    data = data[~data['Dst Port'].str.contains("Dst", na=False)]
    data_train, data_test = train_test_split(data, test_size=0.2)
    data.to_csv(BASE_PATH+"CICIDS2018_all.csv", index=False)
    pd.DataFrame(data_train).to_csv(BASE_PATH+"CICIDS2018_train.csv", index=None)
    pd.DataFrame(data_test).to_csv(BASE_PATH+"CICIDS2018_test.csv", index=None)

def make_DoS_dataset():
    all_data = []
    for f in glob.glob(BASE_PATH+FILE_PATH+"*.csv"):
        data = pd.read_csv(f, index_col=None, header=0)
        all_data.append(data)
    data = pd.concat(all_data, axis=0, ignore_index=True)
    data = data.drop(['Timestamp'], axis=1)
    data = data[~data['Dst Port'].str.contains('Dst', na=False)]
    data = data[(data['Label'].str.contains('Benign')) |
                (data['Label'].str.contains('DoS')) |
                (data['Label'].str.contains('DOS'))]
    data['Label'].replace(['Benign', 'DDOS attack-HOIC',
        'DDOS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP', 'DoS attacks-GoldenEye',
        'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Slowloris'],
        [0, 1, 1, 1, 1, 1, 1, 1], inplace=True)
    data_train, data_test = train_test_split(data, test_size=0.2)
    data_train, data_val = train_test_split(data_train, test_size=0.2)
    data.to_csv(BASE_PATH+"CICIDS2018_all_dos.csv", index=False)
    pd.DataFrame(data_test).to_csv(BASE_PATH+"CICIDS2018_test_dos.csv", index=None)
    pd.DataFrame(data_train).to_csv(BASE_PATH+"CICIDS2018_train_dos.csv", index=None)
    pd.DataFrame(data_val).to_csv(BASE_PATH+"CICIDS2018_val_dos.csv", index=None)

def make_small_dataset():
    all_data = []
    for f in glob.glob(BASE_PATH+FILE_PATH+"*.csv"):
        data = pd.read_csv(f, index_col=None, header=0)
        all_data.append(data)
    data = pd.concat(all_data, axis=0, ignore_index=True)
    data = data.drop(['Timestamp'], axis=1)
    data = data[~data['Dst Port'].str.contains("Dst", na=False)]
    data_train, data_test = train_test_split(data, test_size=0.1)
    pd.DataFrame(data_test).to_csv(BASE_PATH+"CICIDS2018_small.csv", index=None)

def make_small_DoS_dataset():
    data = pd.read_csv(BASE_PATH+"CICIDS2018_small.csv")
    data = data[(data['Label'].str.contains('Benign')) |
                (data['Label'].str.contains('DoS')) |
                (data['Label'].str.contains('DOS'))]
    data['Label'].replace(['Benign', 'DDOS attack-HOIC',
        'DDOS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP', 'DoS attacks-GoldenEye',
        'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Slowloris'],
        [0, 1, 1, 1, 1, 1, 1, 1], inplace=True)
    data_train, data_test = train_test_split(data, test_size=0.2)
    data_train.to_csv(BASE_PATH+"CICIDS2018_small_dos_train.csv", index=False)
    data_test.to_csv(BASE_PATH+"CICIDS2018_small_dos_test.csv", index=False)


if __name__ == "__main__":
    #drop_columns()
    #make_dataset()
    #make_DoS_dataset()
    #make_small_dataset()
    make_small_DoS_dataset()