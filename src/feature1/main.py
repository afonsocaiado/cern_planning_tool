import pandas as pd
import numpy as np

import suggest
import evaluate

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

original_data_with_clusters = pd.read_csv('..\..\data\labeled\\raw.csv', encoding='unicode_escape')
activities = pd.read_csv('..\..\data\\tables\\activity.csv', encoding='unicode_escape')

# Set the columns for which you want to make suggestions
suggest_columns = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE']
relevant_columns = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']

# Data treatment for knn
activities_relevant = activities.loc[:, relevant_columns]

# TESTING FOR A SINGLE INSTANCE
# # create a new pandas entry with the same columns as your original data
# new_entry = pd.DataFrame(columns=['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME'])
# # input the values for the new entry
# new_entry.loc[0] = ['EN-AA', 'PEDRO MARTEL (EN-AA-AC)', 'Consolidation & upgrade/Other', None, 'SYSTEM ADMINISTRATORS']

new_entry_json = {
  "ACTIVITY_UUID": "a6bd2703-7d9c-4115-bb45-cea9b30e90b4",
  "TITLE": "CSAM II- WP3 - Hard. Safety Alarm insfr. (MMD, CSAM)- SPS/LHC- Rempl. of General Zone Alarm cabling",
  "GROUP_RESPONSIBLE_NAME": "EN-AA",
  "RESPONSIBLE_WITH_DETAILS": np.nan,
  "ACTIVITY_TYPE_EN": "Consolidation & upgrade/Other",
  "WBS_NODE_CODE": np.nan,
  "FACILITY_NAMES": "LHC Machine, SPS",
  "CREATOR_NAME": "SYSTEM ADMINISTRATORS",
  "LOCATION_INFORMATION": "SPS/LHC",
  "GOAL": "Preventive Maint. - Hard. Safety Alarm insfr. (MMD, CSAM)- SPS&LHC - Rempl. of General Zone Alarm cabling  by fire resistant new one.\nUnder definition what needs to be done with EN/EL.\nDo not eliminate this activity. To be shifted to after LS2 and evaluating if possible during operation.",
  "IMPACT_NOT_DONE": "System obsolete, no spares available",
}

new_activity = {
  "TITLE": np.nan,
  "GROUP_RESPONSIBLE_NAME": "EN-ACE",
  "RESPONSIBLE_WITH_DETAILS": np.nan,
  "ACTIVITY_TYPE_EN": "Safety",
  "WBS_NODE_CODE": "OTHER",
  "CREATOR_NAME": np.nan,
}

# KNN APPROACH
# knn, columns_to_suggest = suggest.get_nearest_neighbours(new_activity, True, 10)
# suggestions = suggest.make_suggestions_knn(knn, columns_to_suggest, True)

# print(knn)
# print(suggestions)

# knn.to_csv('.\\presentation_test.csv', index=False)

# print("\nOrdered suggestions for the missing information on the new activity:\n")
# for key, values in suggestions.items():
#     print(f"{key}:")
#     for idx, value in enumerate(values):
#         print(f"  Suggestion {idx + 1}: {value}")
#     print()

# KNN APPROACHs
# test_data, original_values = evaluate.create_test_data(activities, test_size=200, suggest_columns=suggest_columns)

# metrics_knn = evaluate.evaluate_suggestions_knn(test_data, original_values)

# for k in range(5, 106, 10):
  # mean_metrics_knn = evaluate.evaluate_suggestions_knn_mean(activities, suggest_columns, k, 200, 5)
  # print(f"Metrics for knn method with {k} clusters: '{mean_metrics_knn}'")

mean_metrics_knn = evaluate.evaluate_suggestions_knn_mean(activities, suggest_columns, 10, 200, 5, 0.3, 0.7)
print(f"Metrics for knn method with 10 clusters: '{mean_metrics_knn}'")

# for i in range(5, 31, 5):
#   mean_metrics_knn = evaluate.cross_validate_knn(activities, suggest_columns, i, n_splits=10)
#   print(f"Cross-validated metrics for knn method with {i} clusters: '{mean_metrics_knn}'")