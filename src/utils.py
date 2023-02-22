import pandas as pd
import sys

from sklearn.preprocessing import LabelEncoder,  MinMaxScaler, StandardScaler
from scipy.stats import zscore

import joblib

def encode(df):

    le = LabelEncoder()

    for col in df.columns:
        le.fit(df[col])
        # joblib.dump(le, "models/" + col + ".joblib")
        df[col] = le.transform(df[col])

    return df

def normalize(df, method):

    if method == "zscore":
        return df.apply(zscore)
    elif method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(df)
        # joblib.dump(scaler, "models/minmax.joblib")
        return scaler.transform(df)
    elif method == "standard":
        scaler = StandardScaler()
        scaler.fit(df)
        return scaler.transform(df)
    else:
        print("Non exsiting normalizing method")
        sys.exit(1)

def remove_nans(df):

    for column in df.columns:
        if df[column].dtype == 'float64':
            df[column] = df[column].fillna(0.00)
        else:
            df[column] = df[column].fillna("Unknown")

    return df
