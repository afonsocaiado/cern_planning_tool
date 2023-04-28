import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reading the database
q1 = pd.read_csv('..\data\processed\\q1.csv', encoding='unicode_escape')

# Data Understanding
# printing the top 10 rows
print(q1.head())

# Dataset information
# Dataframe shape
print("Shape: ", q1.shape)

# Info about the DataFrame
print("\nInfo:")
print(q1.info())

# Descriptive statistics of the numerical columns
print("\nDescriptive Statistics:")
print(q1.describe())

# Number of unique values per column
print("\nUnique values:")
print(q1.nunique())

categorical_features = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES', 'PRIORITY_EN']
numerical_columns = ['PREPARATION_DURATION', 'INSTALLATION_DURATION', 'COMMISSIONING_DURATION']

for feature in categorical_features:
    plt.figure()
    sns.countplot(x=feature, data=q1)
    plt.xticks(rotation=90)
    plt.show()


for column in numerical_columns:
    plt.figure()
    sns.histplot(x=column, data=q1, bins=20)
    plt.show()

# # Create a scatter plot
# plt.scatter(q1['ACTIVITY_TYPE_EN'], q1['PREPARATION_DURATION'], alpha=0.5)

# # Add axis labels and title
# plt.xlabel('Activity Type')
# plt.ylabel('Preparation Duration (days)')
# plt.title('Activity Type vs Preparation Duration')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=90)

# # Show the plot
# plt.show()

# Create scatter plot of WBS_NODE_CODE vs INSTALLATION_DURATION
sns.scatterplot(x='WBS_NODE_CODE', y='INSTALLATION_DURATION', data=q1)
plt.show()

# Create box plot of ACTIVITY_TYPE_EN vs PREPARATION_DURATION
sns.boxplot(x='ACTIVITY_TYPE_EN', y='PREPARATION_DURATION', data=q1)
plt.show()
sns.boxplot(x='ACTIVITY_TYPE_EN', y='INSTALLATION_DURATION', data=q1)
plt.show()
sns.boxplot(x='ACTIVITY_TYPE_EN', y='COMMISSIONING_DURATION', data=q1)
plt.show()

# Create histogram of COMMISSIONING_DURATION
sns.histplot(data=q1, x='COMMISSIONING_DURATION', kde=True)
plt.show()