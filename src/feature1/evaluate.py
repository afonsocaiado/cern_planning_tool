import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

import suggest

def evaluate_suggestions_knn(test_data, original_values, k=10, knnw=0.5, nlpw=0.5):
    
    # Initialize lists to store the true labels and the predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate through each row in the test_data DataFrame
    for index, row in test_data.iterrows():
        # Create a DataFrame containing a single row for the current activity
        activity_df = pd.DataFrame(row).T
        activity_json = activity_df.iloc[0].to_dict()
        
        knn, columns_to_suggest = suggest.get_nearest_neighbours(activity_json, True, k, knnw, nlpw)
        suggestions = suggest.make_suggestions_knn(knn, columns_to_suggest)

        # Iterate through the columns and their corresponding suggested values
        for col, suggested_values in suggestions.items():
            # Append the true label for the current column to the true_labels list
            true_labels.append(original_values.loc[index, col])
            # Append the list of suggested values for the current column to the predicted_labels list
            predicted_labels.append(suggested_values)

    # Calculate evaluation metrics based on the true and predicted labels
    metrics = calculate_metrics(true_labels, predicted_labels)
    return metrics

def evaluate_suggestions_knn_mean(df, suggest_columns, k=10, test_size=200, n_iters=5, knnw=0.5, nlpw=0.5):

    precision_values = []
    recall_values = []
    f1_score_values = []

    for i in range(n_iters):

        test_data, original_values = create_test_data(df, test_size, suggest_columns)

        metrics = evaluate_suggestions_knn(test_data, original_values, k, knnw, nlpw)

        print(f"Run {i+1}: precision: {metrics['precision']}, recall: {metrics['recall']}, f1_score: {metrics['f1_score']}")

        precision_values.append(metrics["precision"])
        recall_values.append(metrics["recall"])
        f1_score_values.append(metrics["f1_score"])

    mean_precision = np.mean(precision_values)
    mean_recall = np.mean(recall_values)
    mean_f1_score = np.mean(f1_score_values)

    return {"precision": mean_precision, "recall": mean_recall, "f1_score": mean_f1_score}

def calculate_weights(position, max_position):

    # Calculate a weight for a given position based on the maximum possible position
    return (max_position - position + 1) / max_position

def calculate_metrics(true_labels, predicted_labels):
    
    tp, fp, fn = 0, 0, 0
    total_predictions = 0
    max_position = len(predicted_labels[0])  # Assuming all predicted_labels lists have the same length
    weighted_precision_sum = 0

    for true, predicted in zip(true_labels, predicted_labels):
        position_weight = 0
        if true in predicted:
            tp += 1
            # Calculate position weight for true label found in the list of predicted labels
            position_weight = calculate_weights(predicted.index(true) + 1, max_position)
        else:
            fn += 1
            fp += len(predicted)

        # Accumulate weighted precision for each true/predicted pair
        weighted_precision_sum += position_weight
        total_predictions += 1

    # Calculate the final weighted precision
    weighted_precision = weighted_precision_sum / total_predictions
    # Calculate recall
    recall = tp / (tp + fn)
    # Calculate F1 score
    f1_score = 2 * weighted_precision * recall / (weighted_precision + recall)

    return {'precision': weighted_precision, 'recall': recall, 'f1_score': f1_score}

def create_test_data(df, test_size=100, suggest_columns=None):
    if suggest_columns is None:
        suggest_columns = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE']

    if test_size is None or test_size > len(df):
        test_size = len(df)

    test_df = df.sample(test_size)

    # Keep only the relevant columns + CREATOR_NAME
    test_df = test_df[['TITLE'] + suggest_columns + ['CREATOR_NAME']]
    original_values = test_df.copy()

    for index, row in test_df.iterrows():
        # Randomly select a subset of suggest_columns to remove values from
        n_removed_fields = np.random.randint(1, len(suggest_columns) + 1)
        removed_fields = np.random.choice(suggest_columns, n_removed_fields, replace=False)
        test_df.loc[index, removed_fields] = np.nan

    return test_df, original_values

def cross_validate_knn(df, suggest_columns, k=10, n_splits=5, test_size=200):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    precision_values = []
    recall_values = []
    f1_score_values = []

    for train_index, test_index in kf.split(df):
        train_data, test_data = df.iloc[train_index], df.iloc[test_index]
        original_values = test_data.copy()

        # Create test data with missing values
        test_data, original_values = create_test_data(test_data, test_size, suggest_columns)

        metrics = evaluate_suggestions_knn(test_data, original_values, k)

        precision_values.append(metrics["precision"])
        recall_values.append(metrics["recall"])
        f1_score_values.append(metrics["f1_score"])

    mean_precision = np.mean(precision_values)
    mean_recall = np.mean(recall_values)
    mean_f1_score = np.mean(f1_score_values)

    return {"precision": mean_precision, "recall": mean_recall, "f1_score": mean_f1_score}
