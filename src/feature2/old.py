import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules
from func_timeout import func_timeout, FunctionTimedOut
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.append('..')  # add parent directory to Python path

import utils
import pickle
import suggest
import evaluate

import warnings
warnings.filterwarnings("ignore")

contributions = pd.read_csv('..\..\data\\tables\\contributions.csv', encoding='unicode_escape')
activities = pd.read_csv('..\..\data\labeled\\raw.csv', encoding='unicode_escape')

relevant_columns = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']

test_data = pd.DataFrame([activities.iloc[17]])

# actual_contributions = contributions[(contributions['ACTIVITY_UUID'].isin(test_data['ACTIVITY_UUID'])) & (contributions['PHASE_NAME'] == 'Preparation')]
# print(actual_contributions)

# # SUGGESTION APPROACHES
# new_activity_cluster = suggest.predict_cluster(test_data[relevant_columns])
# print("Predicted Cluster: ", new_activity_cluster)
# suggestions = suggest.most_common_contributions(new_activity_cluster, activities, contributions, "Preparation", False)
# suggestions = suggest.contributions_association_rules("knn", test_data, None, activities, contributions, 100)
# suggestions = suggest.contributions_collaborative_filtering(test_data, new_activity_cluster, activities, contributions, "Preparation", False, 10)

# EVALUATION
# test_data = activities.sample(100)

# # precision, recall, f1score = evaluate_contributions_suggestions("most_common_contributions", test_data, activities, contributions, relevant_columns, True)
# mean_metrics = evaluate_contributions_suggestions_mean("most_common_contributions", activities, contributions, relevant_columns, True)
# print("Most common contributions in cluster: ", mean_metrics)
# mean_metrics = evaluate_contributions_suggestions_mean("most_common_contributions", activities, contributions, relevant_columns, False)
# print("Most common contributions in total: ", mean_metrics)

# # precision, recall, f1score = evaluate_contributions_suggestions("contributions_collaborative_filtering", test_data, activities, contributions, relevant_columns, True)
# mean_metrics = evaluate_contributions_suggestions_mean("contributions_collaborative_filtering", activities, contributions, relevant_columns, True, 10)
# print("10 n Collaborative filtering in cluster: ", mean_metrics)

# for i in range(5, 106, 10):
#     mean_metrics = evaluate_contributions_suggestions_mean("contributions_collaborative_filtering", activities, contributions, relevant_columns, True, i)
#     print(f"With {i} neighbours with cluster: {mean_metrics}")

# mean_metrics = evaluate_contributions_suggestions_mean("contributions_collaborative_filtering", activities, contributions, relevant_columns, False, 10)
# print("10 n Collaborative filtering in total: ", mean_metrics)

# for i in range(5, 106, 10):
#     mean_metrics = evaluate_contributions_suggestions_mean("contributions_collaborative_filtering", activities, contributions, relevant_columns, False, i)
#     print(f"With {i} neighbours no cluster: {mean_metrics}")

# ------------------------------ OLD SUGGEST FUNCTIONS -------------------------------

def combined_suggestions(new_activity_json, k, activities_df, contributions_df, association_rules, min_support=0.05, min_confidence=0.5):
    
    # Find the k-nearest neighbors and their associated contributions
    knn, _ = suggest.get_nearest_neighbours(new_activity_json, k, False)
    knn_suggestions = suggest.contributions_knn(knn)

    # print(knn_suggestions)

    # Generate association rules
    # association_rules = generate_association_rules(activities_df, contributions_df, min_support, min_confidence)
    single_association_rules = association_rules[association_rules['antecedents'].apply(lambda x: len(x) == 1) & association_rules['consequents'].apply(lambda x: len(x) == 1)]

    # print(association_rules)
    # print(single_association_rules)

    # Combine the results from the k-nearest neighbors and association rules
    new_suggestions = pd.DataFrame(columns=['CONTRIBUTION_TYPE', 'ORG_UNIT_CODE'])

    for phase in ['Preparation', 'Installation', 'Commissioning']:
        for _, rule in single_association_rules.iterrows():
            antecedent = next(iter(rule['antecedents']))
            if antecedent[0] == phase:
                for _, suggestion in knn_suggestions[phase].iterrows():
                    if antecedent[1:] == (suggestion['CONTRIBUTION_TYPE'], suggestion['ORG_UNIT_CODE']):
                        consequent = next(iter(rule['consequents']))
                        new_suggestion = pd.Series({'PHASE_NAME': consequent[0], 'CONTRIBUTION_TYPE': consequent[1], 'ORG_UNIT_CODE': consequent[2], 'confidence': rule['confidence'] * 100})
                        new_suggestions = new_suggestions.append(new_suggestion, ignore_index=True)

    for _, row in new_suggestions.iterrows():
        row_to_append = pd.Series({'CONTRIBUTION_TYPE': row['CONTRIBUTION_TYPE'], 'ORG_UNIT_CODE': row['ORG_UNIT_CODE'], 'confidence': row['confidence']})
        knn_suggestions[row['PHASE_NAME']] = knn_suggestions[row['PHASE_NAME']].append(row_to_append, ignore_index=True).sort_values(by='confidence', ascending=False).reset_index(drop=True)

    for phase in knn_suggestions:
        knn_suggestions[phase] = knn_suggestions[phase].drop_duplicates(subset=['CONTRIBUTION_TYPE', 'ORG_UNIT_CODE']).reset_index(drop=True).head(5)

    return knn_suggestions

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

    # make a prediction using the classifier
    prediction = classifier.predict(activity)

    return prediction[0]

def most_common_contributions(new_activity_cluster, activities_df, contributions_df, phase_name, cluster=True, min_count=3): # MOST COMMON CONTRIBUTIONS APPROACH

    if cluster:
        # Filter the activities that belong to the predicted cluster
        activities_df = activities_df[activities_df['CLUSTER'] == new_activity_cluster]

    # Filter contributions by the given phase
    contributions_in_phase = contributions_df[contributions_df['PHASE_NAME'] == phase_name]
    # Merge activities with contributions data
    merged_data = activities_df.merge(contributions_in_phase, on='ACTIVITY_UUID')
    # Group by CONTRIBUTION_TYPE and ORG_UNIT_CODE, and count the occurrences
    grouped_data = merged_data.groupby(['CONTRIBUTION_TYPE', 'ORG_UNIT_CODE']).size().reset_index(name='count')
    # Filter the grouped_data based on the minimum count threshold
    filtered_data = grouped_data[grouped_data['count'] >= min_count]
    # Sort the data by count (in descending order) and get the top 'n' rows
    n = 5  # You can set 'n' to the number of suggestions you want to make
    top_contributions = grouped_data.sort_values(by='count', ascending=False).head(n)

    # Print the most common contributions as suggestions
    # print("Suggested contributions based on the predicted cluster most common contributions:")
    # for index, row in top_contributions.iterrows():
    #     print(f"{index + 1}. {row['CONTRIBUTION_TYPE']} from {row['ORG_UNIT_CODE']} ({row['count']} occurrences)")

    return top_contributions

def contributions_association_rules(method, new_activity, new_activity_cluster, activities_df, contributions_df, k=10): # ASSOCIATION RULES APPROACH
    
    if method == "cluster":
        # Filter the activities that belong to the predicted cluster
        activities_df = activities_df[activities_df['CLUSTER'] == new_activity_cluster]
    elif method == "knn":
        activities_df = suggest.get_nearest_neighbours(activities_df, new_activity, k)

    # Merge activities_in_cluster with contributions data
    merged_data = activities_df.merge(contributions_df, on='ACTIVITY_UUID')

    # Drop rows with missing or NaN values in the relevant columns
    merged_data = merged_data.dropna(subset=['PHASE_NAME', 'CONTRIBUTION_TYPE', 'ORG_UNIT_CODE'])

    # Group by ACTIVITY_UUID and concatenate CONTRIBUTION_TYPE and ORG_UNIT_CODE as a tuple
    grouped_data = merged_data.groupby('ACTIVITY_UUID').apply(lambda x: [(row['PHASE_NAME'], row['CONTRIBUTION_TYPE'], row['ORG_UNIT_CODE']) for _, row in x.iterrows()])
    
    # Transform the data into a transaction format
    transaction_encoder = TransactionEncoder()
    transaction_array = transaction_encoder.fit(grouped_data).transform(grouped_data)
    # Create a DataFrame with the transaction data
    transaction_df = pd.DataFrame(transaction_array, columns=transaction_encoder.columns_)

    # Compute the frequent itemsets using the Apriori algorithm
    min_support = 0.05  # You can adjust this value to control the minimum support threshold
    timeout_seconds = 10  # Set the timeout in seconds

    try:
        frequent_itemsets = func_timeout(timeout_seconds, apriori, args=(transaction_df, min_support, True))
    except FunctionTimedOut:
        return pd.DataFrame(columns=['antecedents', 'consequents', 'confidence', 'support'])

    # frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)
    # Compute the association rules
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

    # # Print the top 'n' association rules
    # n = 5
    # print("Suggested association rules based on the predicted cluster:")
    # for index, row in rules.head(n).iterrows():
    #     print(f"{index + 1}. {set(row['antecedents'])} => {set(row['consequents'])} (confidence: {row['confidence']:.2f}, support: {row['support']:.2f})")

    # Extract the antecedents and consequents
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0] if len(x) > 0 else None)
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0] if len(x) > 0 else None)

    # Keep only the relevant columns in the DataFrame
    rules = rules[['antecedents', 'consequents', 'confidence', 'support']]

    return rules

def contributions_collaborative_filtering(new_activity, new_activity_cluster, activities_df, contributions_df, phase_name, cluster=True, k=10): # COLLABORATIVE FILTERING APPROACH

    if cluster:
        # Filter the activities that belong to the predicted cluster
        activities_df = activities_df[activities_df['CLUSTER'] == new_activity_cluster]

    # Filter contributions by the given phase
    contributions_in_phase = contributions_df[contributions_df['PHASE_NAME'] == phase_name]
    # Define the categorical and numerical features
    categorical_features = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']
    numerical_features = []
    # Create a preprocessor for one-hot encoding categorical features and scaling numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    # Create a pipeline to preprocess the data and perform k-NN
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('knn', NearestNeighbors(n_neighbors=k))
    ])
    # Fit the pipeline on the activities data
    pipeline.fit(activities_df)
    # Find the k nearest neighbors for the new activity
    new_activity_preprocessed = pipeline.named_steps['preprocessor'].transform(new_activity)
    distances, indices = pipeline.named_steps['knn'].kneighbors(new_activity_preprocessed)
    # Get the ACTIVITY_UUIDs of the nearest neighbors
    nearest_neighbors_uuids = activities_df.iloc[indices[0]]['ACTIVITY_UUID']
    # Merge the nearest neighbors with contributions data
    merged_data = contributions_in_phase[contributions_in_phase['ACTIVITY_UUID'].isin(nearest_neighbors_uuids)]
    # Find the most common contributions within the nearest neighbors
    common_contributions = merged_data.groupby(['CONTRIBUTION_TYPE', 'ORG_UNIT_CODE']).size().reset_index(name='count').sort_values(by='count', ascending=False).head(5)
    
    # # Print the most common contributions as suggestions
    # print("Suggested contributions based on the k nearest neighbors:")
    # for index, row in common_contributions.iterrows():
    #     print(f"{index + 1}. {row['CONTRIBUTION_TYPE']} from {row['ORG_UNIT_CODE']} ({row['count']} occurrences)")

    return common_contributions

def contributions_hybrid(method, new_activity, new_activity_cluster, activities_df, contributions_df, k=10):

    # Get KNN suggestions for all phases
    knn_suggestions = {}

    knn = suggest.get_nearest_neighbours(activities_df, new_activity)

    for phase in ['Preparation', 'Installation', 'Commissioning']:
        knn_suggestions[phase] = suggest.contributions_knn(knn, contributions_df, phase)

    # Get association rules suggestions for all phases
    association_rules_suggestions = contributions_association_rules(method, new_activity, new_activity_cluster, activities_df, contributions_df)

    # Combine KNN suggestions with association rules suggestions
    combined_suggestions = {}

    suggestions_to_add = pd.DataFrame({'PHASE_NAME': [], 'CONTRIBUTION_TYPE': [], 'ORG_UNIT_CODE': [], 'count': []})

    for phase in ['Preparation', 'Installation', 'Commissioning']:

        knn_phase_suggestions = knn_suggestions[phase]
        
        rules_phase_suggestions = association_rules_suggestions[association_rules_suggestions['antecedents'].apply(lambda x: x[0] == phase)]

        print(knn_phase_suggestions)
        print(rules_phase_suggestions)

        for index1, rule in rules_phase_suggestions.iterrows():

            for index2, suggestion in knn_phase_suggestions.iterrows():

                if rule['antecedents'][1] == suggestion['CONTRIBUTION_TYPE'] and rule['antecedents'][2] == suggestion['ORG_UNIT_CODE']:
                    
                    new_suggestion = {'PHASE_NAME': rule['consequents'][0], 'CONTRIBUTION_TYPE': rule['consequents'][1], 'ORG_UNIT_CODE': rule['consequents'][2], 'count': np.nan}
                    suggestions_to_add.loc[len(suggestions_to_add)] = new_suggestion

        print(suggestions_to_add)

        # Combine the suggestions for the current phase
        combined = pd.concat([knn_phase_suggestions, rules_phase_suggestions], ignore_index=True)
        combined_suggestions[phase] = combined

    return combined_suggestions

# ------------------------------ OLD EVALUATE FUNCTIONS -------------------------------

def evaluate_contributions_suggestions(method, test_data, activities, contributions, relevant_columns, cluster=True, k=10):

    total_precision = 0
    total_recall = 0
    total_f1score = 0
    num_rows = len(test_data)
    valid_rows = 0  # Initialize a counter for valid rows (with contributions)

    for index, row in test_data.iterrows():

        # Create a DataFrame containing a single row for the current activit
        new_activity = pd.DataFrame(row).T

        # Get the contributions for the Preparation phase
        prep_contributions = contributions[(contributions['ACTIVITY_UUID'].isin(new_activity['ACTIVITY_UUID'])) & (contributions['PHASE_NAME'] == 'Preparation')]
        # Get the contributions for the Installation phase
        install_contributions = contributions[(contributions['ACTIVITY_UUID'].isin(new_activity['ACTIVITY_UUID'])) & (contributions['PHASE_NAME'] == 'Installation')]
        # Get the contributions for the Commissioning phase
        commissioning_contributions = contributions[(contributions['ACTIVITY_UUID'].isin(new_activity['ACTIVITY_UUID'])) & (contributions['PHASE_NAME'] == 'Commissioning')]

        new_activity_cluster = suggest.predict_cluster(new_activity[relevant_columns])

        if method == "most_common_contributions":
            min_count_value = 0
            prep_suggestions = most_common_contributions(new_activity_cluster, activities, contributions, "Preparation", cluster, min_count_value)
            install_suggestions = most_common_contributions(new_activity_cluster, activities, contributions, "Installation", cluster, min_count_value)
            commissioning_suggestions = most_common_contributions(new_activity_cluster, activities, contributions, "Commissioning", cluster, min_count_value)
        elif method == "contributions_collaborative_filtering":
            prep_suggestions = contributions_collaborative_filtering(new_activity, new_activity_cluster, activities, contributions, "Preparation", cluster, k)
            install_suggestions = contributions_collaborative_filtering(new_activity, new_activity_cluster, activities, contributions, "Installation", cluster, k)
            commissioning_suggestions = contributions_collaborative_filtering(new_activity, new_activity_cluster, activities, contributions, "Commissioning", cluster, k)

        prep_precision, prep_recall, prep_f1score = evaluate.weighted_scores(prep_contributions, prep_suggestions)
        install_precision, install_recall, install_f1score = evaluate.weighted_scores(install_contributions, install_suggestions)
        commissioning_precision, commissioning_recall, commissioning_f1score = evaluate.weighted_scores(commissioning_contributions, commissioning_suggestions)

        # Check if any of the phases returned None values
        if prep_precision is not None and install_precision is not None and commissioning_precision is not None:
            valid_rows += 1

            average_precision = (prep_precision + install_precision + commissioning_precision) / 3
            average_recall = (prep_recall + install_recall + commissioning_recall) / 3
            average_f1score = (prep_f1score + install_f1score + commissioning_f1score) / 3

            total_precision += average_precision
            total_recall += average_recall
            total_f1score += average_f1score

    
    # Compute the final averages by dividing the sums by the number of rows
    final_average_precision = total_precision / num_rows
    final_average_recall = total_recall / num_rows
    final_average_f1score = total_f1score / num_rows

    return final_average_precision, final_average_recall, final_average_f1score

def evaluate_contributions_suggestions_mean(method, activities, contributions, relevant_columns, cluster=True, k=10, n_iters=5):

    precision_values = []
    recall_values = []
    f1_score_values = []

    for i in range(n_iters):

        test_data = activities.sample(100)

        precision, recall, f1score = evaluate_contributions_suggestions(method, test_data, activities, contributions, relevant_columns, cluster, k)

        # print(f"Run {i+1}: precision: {precision}, recall: {recall}, f1_score: {f1score}")

        precision_values.append(precision)
        recall_values.append(recall)
        f1_score_values.append(f1score)

    mean_precision = np.mean(precision_values)
    mean_recall = np.mean(recall_values)
    mean_f1_score = np.mean(f1_score_values)
