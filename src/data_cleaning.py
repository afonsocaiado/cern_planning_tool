import pandas as pd

# load data from generated CSV file
created_q1 = pd.read_csv('..\data\processed\\created_q1.csv', encoding='unicode_escape')

# check for NaNs in the entire dataframe
print(created_q1.isna().sum())

# to start off we want to remove Nans from ACTIVITY_TYPE_EN, WBS_NODE_CODE, and FACILITY_NAMES as they are the ones we will start clustering with
# as all 3 are categorical text values, we will start of by creating an additional 'Unknown' category for each
created_q1["ACTIVITY_TYPE_EN"] = created_q1["ACTIVITY_TYPE_EN"].fillna("Unknown")
created_q1["WBS_NODE_CODE"] = created_q1["WBS_NODE_CODE"].fillna("Unknown")
created_q1["FACILITY_NAMES"] = created_q1["FACILITY_NAMES"].fillna("Unknown")
# check for NaNs in the new dataframe
print("\n", created_q1.isna().sum())


created_q1.to_csv('..\data\processed\clean_q1.csv', index=False)
