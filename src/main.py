import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# The Dataset Acquisition has been done before with data_acquisition.py
df = pd.read_csv('..\data\processed\\q1.csv', encoding='unicode_escape')

# DATA SPLITTING
# Split the DataFrame into a train set (70%) and a test set (30%)
train, test = train_test_split(df, test_size=0.1, random_state=0)

print("Train Shape: ", train.shape)
