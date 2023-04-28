import pandas as pd
import numpy as np

import sys
sys.path.append('..')  # add parent directory to Python path

import utils
import suggest
import evaluate

import pickle

original_data_with_clusters = pd.read_csv('..\..\data\labeled\\raw.csv', encoding='unicode_escape')
activities = pd.read_csv('..\..\data\\tables\\activity.csv', encoding='unicode_escape')

# Set the columns for which you want to make suggestions
suggest_columns = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE']
relevant_columns = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']

# TESTING FOR A SINGLE INSTANCE
# # create a new pandas entry with the same columns as your original data
# new_entry = pd.DataFrame(columns=['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME'])
# # input the values for the new entry
# new_entry.loc[0] = ['EN-AA', 'PEDRO MARTEL (EN-AA-AC)', 'Consolidation & upgrade/Other', None, 'SYSTEM ADMINISTRATORS']

# MOST COMMON APPROACH
# cluster = suggest.predict_cluster(new_entry)
# print(cluster)
# suggestions = suggest.make_suggestions(new_entry, cluster, original_data_with_clusters, suggest_columns)
# suggestions = suggest.make_suggestions_no_cluster(new_entry, original_data_with_clusters, suggest_columns)

# new_entry, new_entry_original = evaluate.create_test_data(activities, 1)
# print(new_entry.head())

# SUGGESTION EVALUATION

# MOST COMMON APPROACH
# test_data, original_values = evaluate.create_test_data(original_data_with_clusters, test_size=200, suggest_columns=suggest_columns)

# metrics = evaluate_suggestions(test_data, original_values, original_data_with_clusters, suggest_columns, use_clustering=True)
# mean_metrics_cluster = evaluate_suggestions_mean(original_data_with_clusters, suggest_columns, use_clustering=True)
# print(f"Metrics for clustering method: '{mean_metrics_cluster}'")
# metrics_no_cluster = evaluate_suggestions(test_data, original_values, original_data_with_clusters, suggest_columns, use_clustering=False)
# mean_metrics_no_cluster = evaluate_suggestions_mean(original_data_with_clusters, suggest_columns, use_clustering=False)
# print(f"Metrics for no clustering method: '{mean_metrics_no_cluster}'")

# ------------------------------ OLD SUGGEST FUNCTIONS -------------------------------

def predict_cluster(activity):

    for col in activity.columns:
        activity[col] = activity[col].astype('object')

    activity = utils.remove_nans(activity)

    # Load encoders from disk
    with open('..\clustering\encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    # Load encoders from disk
    with open('..\supervised\classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)

    for col, encoder in encoders.items():
        activity[col] = encoder.transform(activity[col])

    # List of features used for training the classifier
    relevant_features = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']

    # Select only the relevant features
    activity_relevant = activity[relevant_features]

    # make a prediction using the classifier
    prediction = classifier.predict(activity_relevant)

    return prediction[0]

def get_suggestions(column, max_list_size=5):
    return column.dropna().value_counts().nlargest(max_list_size).index.tolist()

def make_suggestions(activity_relevant, cluster, df, suggest_columns):

    empty_fields = {}

    cluster_df = df[df['CLUSTER'] == cluster]

    cluster_df = cluster_df[suggest_columns]

    for col in suggest_columns:
        if pd.isna(activity_relevant[col].iloc[0]):
            empty_fields[col] = get_suggestions(cluster_df[col])

    # print("Suggested values based on the most common values within the cluster:")
    # for col, values in empty_fields.items():
    #     print(f"\nSuggestions for '{col}':")
    #     for index, value in enumerate(values):
    #         print(f"{index + 1}. {value} ({cluster_df[col].value_counts()[value]} occurrences)")

    return empty_fields

def make_suggestions_no_cluster(activity, df, suggest_columns):
    
    suggestions = {}

    for col in suggest_columns:
        if pd.isna(activity[col].iloc[0]):
            suggestions[col] = get_suggestions(df[col])

    # print("Suggested values based on the most common values within the entire dataset:")
    # for col, values in suggestions.items():
    #     print(f"\nSuggestions for '{col}':")
    #     for index, value in enumerate(values):
    #         print(f"{index + 1}. {value} ({df[col].value_counts()[value]} occurrences)")

    return suggestions

# ------------------------------ OLD EVALUATE FUNCTIONS ------------------------------

def evaluate_suggestions(test_data, original_values, df, suggest_columns, use_clustering=True):

    # Initialize lists to store the true labels and the predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate through each row in the test_data DataFrame
    for index, row in test_data.iterrows():
        # Create a DataFrame containing a single row for the current activity
        activity_df = pd.DataFrame(row).T
        if use_clustering:
            # Predict the cluster for the current activity
            cluster = suggest.predict_cluster(activity_df)
            # Generate suggestions for the current activity based on its predicted cluster and available information
            suggestions = suggest.make_suggestions(activity_df, cluster, df, suggest_columns)
        else: 
            suggestions = suggest.make_suggestions_no_cluster(activity_df, df, suggest_columns)

        # Iterate through the columns and their corresponding suggested values
        for col, suggested_values in suggestions.items():
            # Append the true label for the current column to the true_labels list
            true_labels.append(original_values.loc[index, col])
            # Append the list of suggested values for the current column to the predicted_labels list
            predicted_labels.append(suggested_values)

    # Calculate evaluation metrics based on the true and predicted labels
    metrics = evaluate.calculate_metrics(true_labels, predicted_labels)
    return metrics

def evaluate_suggestions_mean(df, suggest_columns, use_clustering=True, n_iters=5):
    
    precision_values = []
    recall_values = []
    f1_score_values = []

    for i in range(n_iters):

        test_data, original_values = evaluate.create_test_data(df, test_size=200, suggest_columns=suggest_columns)

        metrics = evaluate_suggestions(test_data, original_values, df, suggest_columns, use_clustering)
        
        print(f"Run {i+1}: precision: {metrics['precision']}, recall: {metrics['recall']}, f1_score: {metrics['f1_score']}")

        precision_values.append(metrics["precision"])
        recall_values.append(metrics["recall"])
        f1_score_values.append(metrics["f1_score"])

    mean_precision = np.mean(precision_values)
    mean_recall = np.mean(recall_values)
    mean_f1_score = np.mean(f1_score_values)

    return {"precision": mean_precision, "recall": mean_recall, "f1_score": mean_f1_score}
