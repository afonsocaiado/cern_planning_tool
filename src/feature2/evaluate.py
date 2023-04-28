import pandas as pd
import numpy as np

import suggest

def evaluate_contributions_suggestions(test_data, activities, contributions, association_rules, k=10, min_support=0.05, min_confidence=0.5):

    total_precision = 0
    total_recall = 0
    total_f1score = 0
    num_rows = len(test_data)
    valid_rows = 0  # Initialize a counter for valid rows (with contributions)

    # association_rules = suggest.generate_association_rules(activities, contributions, min_support, min_confidence)

    for index, row in test_data.iterrows():

        # Create a DataFrame containing a single row for the current activit
        new_activity = pd.DataFrame(row).T
        new_activity_json = new_activity.iloc[0].to_dict()

        # Get the contributions for the Preparation phase
        prep_contributions = contributions[(contributions['ACTIVITY_UUID'].isin(new_activity['ACTIVITY_UUID'])) & (contributions['PHASE_NAME'] == 'Preparation')]
        # Get the contributions for the Installation phase
        install_contributions = contributions[(contributions['ACTIVITY_UUID'].isin(new_activity['ACTIVITY_UUID'])) & (contributions['PHASE_NAME'] == 'Installation')]
        # Get the contributions for the Commissioning phase
        commissioning_contributions = contributions[(contributions['ACTIVITY_UUID'].isin(new_activity['ACTIVITY_UUID'])) & (contributions['PHASE_NAME'] == 'Commissioning')]

        knn, nan_columns = suggest.get_nearest_neighbours(new_activity_json, k, False)
        suggestions = suggest.contributions_knn(knn)

        # suggestions = suggest.combined_suggestions(new_activity_json, k, activities, contributions, association_rules)

        # prep_suggestions = suggest.contributions_knn_phase(knn, "Preparation")
        # install_suggestions = suggest.contributions_knn_phase(knn, "Installation")
        # commissioning_suggestions = suggest.contributions_knn_phase(knn, "Commissioning")

        prep_suggestions = suggestions["Preparation"]
        install_suggestions = suggestions["Installation"]
        commissioning_suggestions = suggestions["Commissioning"]

        prep_precision, prep_recall, prep_f1score = weighted_scores(prep_contributions, prep_suggestions)
        install_precision, install_recall, install_f1score = weighted_scores(install_contributions, install_suggestions)
        commissioning_precision, commissioning_recall, commissioning_f1score = weighted_scores(commissioning_contributions, commissioning_suggestions)

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

def evaluate_contributions_suggestions_mean(activities, contributions, k=10, min_support=0.03, min_confidence=0.5, n_iters=5):

    precision_values = []
    recall_values = []
    f1_score_values = []

    association_rules = suggest.generate_association_rules(activities, contributions, min_support, min_confidence)

    for i in range(n_iters):

        test_data = activities.sample(100)

        precision, recall, f1score = evaluate_contributions_suggestions(test_data, activities, contributions, association_rules, k, min_support, min_confidence)

        print(f"Run {i+1}: precision: {precision}, recall: {recall}, f1_score: {f1score}")

        precision_values.append(precision)
        recall_values.append(recall)
        f1_score_values.append(f1score)

    mean_precision = np.mean(precision_values)
    mean_recall = np.mean(recall_values)
    mean_f1_score = np.mean(f1_score_values)

    return {"precision": mean_precision, "recall": mean_recall, "f1_score": mean_f1_score}

def calculate_weights(rank, n=5):
    return 1 / rank

def weighted_scores(actual, suggestions):

    actual["combined"] = actual["CONTRIBUTION_TYPE"].astype(str) + "_" + actual["ORG_UNIT_CODE"].astype(str)
    suggestions["combined"] = suggestions["CONTRIBUTION_TYPE"].astype(str) + "_" + suggestions["ORG_UNIT_CODE"].astype(str)

    actual_set = set(actual["combined"].tolist())
    weighted_precision = 0
    weighted_recall = 0
    total_weights = 0
    matches = 0

    # Check if there are no actual contributions
    if len(actual_set) == 0:
        # Return neutral scores (1.0 for precision and recall, and F1 score)
        return 1.0, 1.0, 1.0
        # # Check if the model suggested any contributions
        # if len(suggestions) > 0:
        #     # Introduce a penalty for suggesting contributions when there are none
        #     return 0, 0, 0
        # else:
        #     # Return None values to skip this case
        #     return None, None, None

    for idx, row in suggestions.iterrows():
        weight = calculate_weights(idx + 1)
        total_weights += weight

        if row["combined"] in actual_set:
            matches += 1
            weighted_precision += weight
            weighted_recall += 1

    if matches == 0:
        return (0, 0, 0)

    weighted_precision /= total_weights
    weighted_recall /= len(actual)
    weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)

    return weighted_precision, weighted_recall, weighted_f1

