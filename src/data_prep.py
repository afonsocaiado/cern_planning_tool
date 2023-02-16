import pandas as pd
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def encode(df):

    # Categorical fields
    label_encoders = {}
    for column in df:
        label_encoders[column] = LabelEncoder()
        
    # transform the categorical features to numerical using the label encoders
    for column, label_encoder in label_encoders.items():
        if column == "TITLE":
            continue
        else:
            df[column] = label_encoder.fit_transform(df[column])

    for column in df:
        if column == "TITLE":
            # convert the text column to numerical data using TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english')
            text_features = vectorizer.fit_transform(df[column])

            # combine the text features with the numerical columns in your dataframe
            numerical_features = df.drop(column, axis=1)
            df = pd.concat([pd.DataFrame(text_features.toarray()), numerical_features], axis=1)
        else:
            continue

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
