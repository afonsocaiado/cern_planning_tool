import pandas as pd
import sys

from sklearn.preprocessing import LabelEncoder,  MinMaxScaler
from scipy.stats import zscore

def encode(df):

    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])

    return df

def normalize(df, method):

    if method == "zscore":
        return df.apply(zscore)
    elif method == "minmax":
        scaler = MinMaxScaler()
        return scaler.fit_transform(df)
    else:
        print("Non exsiting normalizing method")
        sys.exit(1)

def remove_nans(df):

    for column in df.columns:
        if df[column].dtype == 'float64':
            df[column] = df[column].fillna(0.00)
        else:
            df[column] = df[column].fillna("Unknown")
