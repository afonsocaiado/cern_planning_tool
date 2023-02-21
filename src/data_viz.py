import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reading the database
q1 = pd.read_csv('..\data\processed\\q1.csv', encoding='unicode_escape')

# printing the top 10 rows
print(q1.head(10))

categorical_features = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES', 'PRIORITY_EN']
numerical_columns = ['PREPARATION_DURATION', 'INSTALLATION_DURATION', 'COMMISSIONING_DURATION']

# for feature in categorical_features:
#     plt.figure()
#     sns.countplot(x=feature, data=q1)
#     plt.xticks(rotation=90)
#     plt.show()


# for column in numerical_columns:
#     plt.figure()
#     sns.histplot(x=column, data=q1, bins=20)
#     plt.show()

# # Create a scatter plot
# plt.scatter(df['ACTIVITY_TYPE_EN'], df['PREPARATION_DURATION'], alpha=0.5)

# # Add axis labels and title
# plt.xlabel('Activity Type')
# plt.ylabel('Preparation Duration (days)')
# plt.title('Activity Type vs Preparation Duration')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=90)

# # Show the plot
# plt.show()