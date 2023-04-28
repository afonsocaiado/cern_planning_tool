import pandas as pd
import numpy as np
import sys

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from scipy.stats import zscore

import pickle

def encode(df, method):

    encoders = {}

    for col in df.columns:

        if not np.issubdtype(df[col].dtype, np.number):

            if method == "label":     
                encoder = LabelEncoder()
                encoder.fit(df[col])
                df[col] = encoder.transform(df[col])
                encoders[col] = encoder
            elif method == "onehot":
                encoder = OneHotEncoder()
                encoded_cols = encoder.fit_transform(df[col].values.reshape(-1, 1)).toarray()
                encoded_df = pd.DataFrame(encoded_cols, columns=[f"{col}_{i}" for i in range(encoded_cols.shape[1])])
                df = pd.concat([df, encoded_df], axis=1).drop(col, axis=1)
                encoders[col] = encoder
            else:
                print("Non exsiting normalizing method")
                sys.exit(1)

    return df, encoders

def normalize(df, method):

    if method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(df)
        df = scaler.transform(df)
        return df
    elif method == "zscore":
        scaler = StandardScaler()
        scaler.fit(df)
        df = scaler.transform(df)
        return df
    else:
        print("Non exsiting normalizing method")
        sys.exit(1)

def remove_nans(df):

    df_copy = df.copy()

    for column in df_copy.columns:
        if df_copy[column].dtype == 'float64':
            df_copy[column] = df_copy[column].fillna(-1)
        else:
            df_copy[column] = df_copy[column].fillna("Unknown")

    return df_copy

def jaccard_dissim_label(a, b, **__):
    """Jaccard dissimilarity function for label encoded variables"""
    if np.isnan(a.astype('float64')).any() or np.isnan(b.astype('float64')).any():
        raise ValueError("Missing values detected in Numeric columns.")
    intersect_len = np.empty(len(a), dtype=int)
    union_len = np.empty(len(a), dtype=int)
    ii = 0
    for row in a:
        intersect_len[ii] = len(np.intersect1d(row, b))
        union_len[ii] = len(np.unique(row)) + len(np.unique(b)) - intersect_len[ii]
        ii += 1
    if (union_len == 0).any():
        raise ValueError("Insufficient Number of data since union is 0")
    return 1 - intersect_len / union_len

def jaccard_dissim_matrix(data):
    n = len(data)
    dissimilarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            intersect_len = len(np.intersect1d(data[i], data[j]))
            union_len = len(np.unique(data[i])) + len(np.unique(data[j])) - intersect_len
            dissimilarity_matrix[i, j] = 1 - intersect_len / union_len

    return dissimilarity_matrix

# def jaccard_dissim_silhouette(u, v, one_hot_indices):
#     intersect = np.intersect1d(u, v)
#     union = np.unique(np.concatenate([u, v]))

#     # Adjust for one-hot encoded variables
#     for idx in one_hot_indices:
#         intersect[idx] = min(u[idx], v[idx])
#         union[idx] = max(u[idx], v[idx])

#     intersect_len = len(intersect)
#     union_len = len(union)

#     return 1 - intersect_len / union_len

def jaccard_dissim_silhouette(u, v):
    """Jaccard dissimilarity function for label encoded variables, for individual data points"""
    intersect_len = len(np.intersect1d(u, v))
    union_len = len(np.unique(u)) + len(np.unique(v)) - intersect_len
    return 1 - intersect_len / union_len

def remove_outliers(df):

    df_copy = df.copy()

    numerical_vars = ['PREPARATION_DURATION', 'INSTALLATION_DURATION', 'COMMISSIONING_DURATION']

    # remove outliers using Z-score method
    df_copy = df_copy[(np.abs(zscore(df_copy[numerical_vars])) < 3).all(axis=1)]

    return df_copy

def prep_data(df):

    df_copy = df.copy()
    
    categorical_vars = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']

    # encode categorical values
    df_categ = df_copy[categorical_vars]
    df_categ, encoders = encode(df_categ, "label")

    # Save encoders to disk
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    for column in categorical_vars:
        df_copy[column] = df_categ[column]

    return df_copy