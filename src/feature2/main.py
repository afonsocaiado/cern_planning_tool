import pandas as pd
import numpy as np

import suggest
import evaluate
import time

from flask import jsonify

import warnings
warnings.filterwarnings("ignore")

contributions = pd.read_csv('..\..\data\\tables\\contributions.csv', encoding='unicode_escape')
activities = pd.read_csv('..\..\data\labeled\\raw.csv', encoding='unicode_escape')

relevant_columns = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']

new_entry_json = {
  "ACTIVITY_UUID": "a6bd2703-7d9c-4115-bb45-cea9b30e90b4",
  "TITLE": "CSAM II- WP3 - Hard. Safety Alarm insfr. (MMD, CSAM)- SPS/LHC- Rempl. of General Zone Alarm cabling",
  "GROUP_RESPONSIBLE_NAME": "EN-AA",
  "RESPONSIBLE_WITH_DETAILS": "PEDRO MARTEL (EN-AA-AC)",
  "ACTIVITY_TYPE_EN": "Consolidation & upgrade/Other",
  "WBS_NODE_CODE": "ACCONS",
  "FACILITY_NAMES": "LHC Machine, SPS",
  "PRIORITY_EN": "4. Approved projects",
  "CREATOR_NAME": "SYSTEM ADMINISTRATORS"
}

new_activity = {
  "TITLE": "Injectors De-cabling phase 2 project - PS ring (2 more octants)",
  "GROUP_RESPONSIBLE_NAME": "EN-ACE",
  "RESPONSIBLE_WITH_DETAILS": "FERNANDO BALTASAR DOS SANTOS PEDR (EN-ACE-OSS)",
  "ACTIVITY_TYPE_EN": "Safety",
  "WBS_NODE_CODE": "OTHER",
  "CREATOR_NAME": "FERNANDO BALTASAR DOS SANTOS PEDR (EN-ACE-OSS)"
}

new_contributions_json = [
    {
      "PHASE_NAME": "Preparation",
      "CONTRIBUTION_TYPE": "Signal cabling",
      "ORG_UNIT_CODE": "EN-EL"
    },
    {
      "PHASE_NAME": "Commissioning",
      "CONTRIBUTION_TYPE": "Other",
      "ORG_UNIT_CODE": "HSE-OHS"
    }
]

new_activity_contributions_json = [
    {
      "PHASE_NAME": "Preparation",
      "CONTRIBUTION_TYPE": "Other",
      "ORG_UNIT_CODE": "HSE-RP"
    }
]

test_data = pd.DataFrame([activities.iloc[17]])

new_entry_df = pd.DataFrame.from_dict(new_entry_json, orient='index').T

# actual_contributions = contributions[(contributions['ACTIVITY_UUID'].isin(test_data['ACTIVITY_UUID'])) & (contributions['PHASE_NAME'] == 'Preparation')]
# print(actual_contributions)

# # SUGGESTION APPROACHES

# knn, nan_columns = suggest.get_nearest_neighbours(new_entry_json, 10, False)
# prep_suggestions = suggest.contributions_knn_phase(knn, "Preparation")
# install_suggestions = suggest.contributions_knn_phase(knn, "Installation")
# com_suggestions = suggest.contributions_knn_phase(knn, "Commissioning")

# knn, nan_columns = suggest.get_nearest_neighbours(new_activity, 10, False)
# suggestions = suggest.contributions_knn(knn)
# print(suggestions)

# # Call the contributions_knn() function
# suggestions = contributions_knn(knn)

# # Print the suggestions for each phase
# for phase_name, phase_suggestions in suggestions.items():
#     print(f"\nTop 5 suggested contributions for {phase_name}:")
#     print(phase_suggestions.to_string(index=False))


# rules = suggest.generate_association_rules(activities, contributions, 0.03, 0.5)
# print(rules)

# suggestions = suggest.combined_suggestions(new_activity, new_activity_contributions_json, 10)
# print(suggestions)

# # Print the suggestions for each phase
# for phase_name, phase_suggestions in suggestions.items():
#     print(f"\nTop 5 suggested contributions for {phase_name}:")
#     print(phase_suggestions.to_string(index=False))

# EVALUATION
# test_data = activities.sample(100)

# precision, recall, f1score = evaluate.evaluate_contributions_suggestions(test_data, activities, contributions, relevant_columns)
# start_time = time.time()
# mean_metrics = evaluate.evaluate_contributions_suggestions_mean(activities, contributions, 10)
# print("10 knn: ", mean_metrics)
# end_time = time.time()

# elapsed_time = end_time - start_time
# print(f"The function took {elapsed_time} seconds to run.")

# for support in range(2, 6):
#   print(support * 0.01)
#   mean_metrics = evaluate.evaluate_contributions_suggestions_mean(activities, contributions, 10, support*0.01, 0.5) # 0.03 forward makes no diff
#   print(f"With {support*0.01} min support: {mean_metrics}")

# for confidence in range(2, 6):
#   mean_metrics = evaluate.evaluate_contributions_suggestions_mean(activities, contributions, 10, 0.03, confidence*0.1) # 0.5 is good, then check weights 
#   print(f"With {confidence*0.01} min confidence: {mean_metrics}")

# for i in range(5, 36, 10):
#     mean_metrics = evaluate.evaluate_contributions_suggestions_mean(activities, contributions, i)
#     print(f"With {i} neighbours: {mean_metrics}")

# print(mean_metrics)

print(activities.shape)
print(contributions.shape)
