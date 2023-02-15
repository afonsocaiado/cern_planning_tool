import sys

def normalize(df, method):

    if method == "zscore":
        return (df - df.mean()) / df.std()
    else:
        print("Non exsiting normalizing method")
        sys.exit(1)