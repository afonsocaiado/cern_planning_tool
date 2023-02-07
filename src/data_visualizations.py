import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

activity_list = pd.read_csv('..\data\\activity_list.csv', encoding='unicode_escape')

# Dataset information
# Dataframe shape
print("Shape: ", activity_list.shape)

# Info about the DataFrame
print("\nInfo:")
print(activity_list.info())

# Descriptive statistics of the numerical columns
print("\nDescriptive Statistics:")
print(activity_list.describe())

# Number of unique values per column
print("\nUnique values:")
print(activity_list.nunique())

# PLAN ID, UUID, NAME, VERSION ID, VERSION UUID, VERSION all the same = probably irrelevant
