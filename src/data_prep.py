import sys

from sklearn.preprocessing import LabelEncoder

def encode(df):

    label_encoders = {}
    for column in df:
        label_encoders[column] = LabelEncoder()
        
    # transform the categorical features to numerical using the label encoders
    for column, label_encoder in label_encoders.items():
        df[column] = label_encoder.fit_transform(df[column])

    return df

def normalize(df, method):

    if method == "zscore":
        return (df - df.mean()) / df.std()
    else:
        print("Non exsiting normalizing method")
        sys.exit(1)

def remove_nans(df):
    for column in df.columns:
        df[column] = df[column].fillna("Unknown")
