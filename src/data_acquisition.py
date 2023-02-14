import pandas as pd
import numpy as np

activity_list = pd.read_csv('..\data\original\\activity_list.csv', encoding='unicode_escape')
activity = pd.read_csv('..\data\original\\activity.csv', encoding='unicode_escape')
phase = pd.read_csv('..\data\original\phase.csv', encoding='unicode_escape')
hl_schedulable_phase = pd.read_csv('..\data\original\hl_schedulable_phase.csv', encoding='unicode_escape')
period = pd.read_csv('..\data\original\period.csv', encoding='unicode_escape')
hl_sched_prop_period = pd.read_csv('..\data\original\hl_sched_prop_period.csv', encoding='unicode_escape')
hl_sched_acce_period = pd.read_csv('..\data\original\hl_sched_acce_period.csv', encoding='unicode_escape')

# Dataset information
# Dataframe shape
# print("Shape: ", hl_sched_prop_period.shape)

# Info about the DataFrame
# print("\nInfo:")
# print(activity_list.info())

# Descriptive statistics of the numerical columns
# print("\nDescriptive Statistics:")
# print(activity_list.describe())

# Number of unique values per column
# print("\nUnique values:")
# print(activity_list.nunique())

# ACTIVITY LIST
activity_list = activity_list.drop(columns=['PLAN_ID', 'PLAN_UUID', 'PLAN_NAME', 'PLAN_VERSION_ID', 'PLAN_VERSION_UUID', 'PLAN_VERSION']) # PLAN ID, UUID, NAME, VERSION ID, VERSION UUID, VERSION all the same = probably irrelevant
# Removing UUIDS
activity_list = activity_list.drop(columns=['GROUP_RESPONSIBLE_UUID', 'RESPONSIBLE_UUID', 'ACTIVITY_TYPE_UUID', 'WBS_NODE_UUID', 'FACILITIES', 'GROUP_CONTRIBUTIONS', 'PRIORITY_UUID', 'CREATOR_UUID' ])
activity_list = activity_list.drop(columns=['ACTIVITY_TYPE_FR', 'PRIORITY_FR', 'STATUS', 'RESPONSIBLE_ORG_UNIT', 'RESPONSIBLE_PERSON_ID', 'RESPONSIBLE_FULL_NAME', 'HAS_DATA_QUALITY', 'HAS_NOT_OK_COMMENT_FOLLOW_UP', 'LAST_UPDATE_DATE'])
# Contributions aren't for right now
activity_list = activity_list.drop(columns=['GROUP_CONTRIBUTION_NAMES', 'GROUP_CONTRIBUTIONS_NUM', 'VALID_GROUP_CONTRIBUTIONS_NUM', 'VALID_GROUP_CONTRIBUTIONS_PCT', 'ALLOCATED_GROUP_CONTRIB_NUM', 'CLEARED_GROUP_CONTRIB_NUM', 'DEALLOCATED_GROUP_CONTRIB_NUM', 'ALLOCATED_GROUP_CONTRIB_PCT', 'GROUP_CONTRIB_RES', 'UNDEFINED_GROUP_CONTRIB_NUM' ])
activity_list = activity_list.drop(columns=['STATUS_REASON_UUID', 'STATUS_REASON', 'STATUS_COMMENT', 'PARENT_ACTIVITY_STATUS']) # STATUS_REASON_UUID, STATUS_REASON, STATUS_COMMENT doesnt seem to be relevant either, only 2 values and isn't crucial for creation of a new activity
activity_list = activity_list.drop(columns=['RESPONSIBLE_SECTION_UUID', 'RESPONSIBLE_GROUP_UUID', 'RESPONSIBLE_DEPARTMENT_UUID'])

# ACTIVITY
activity = activity.filter(items=['ID', 'LOCATION_INFORMATION', 'GOAL', 'IMPACT_NOT_DONE'])

# SCHEDULE
# PHASE
phase = phase.filter(items=['ID', 'NAME_EN'])
phase.rename(columns={'ID': 'PHASE_UUID', 'NAME_EN': 'PHASE_NAME'}, inplace=True)
# HL SCHEDULABLE PHASE
hl_schedulable_phase = hl_schedulable_phase.drop(columns=['SCHEDULE_ID'])
hl_schedulable_phase.rename(columns={'ID': 'SCHEDULABLE_PHASE_UUID', 'AMOUNT': 'PHASE_AMOUNT', 'DURATION': 'PHASE_DURATION'}, inplace=True)
# PERIOD
period = period.drop(columns=['PERIOD_ID', 'PLAN_UUID', 'START_DATE', 'END_DATE', 'ACTIVE', 'PERIOD_ORDER', 'PARENT_UUID'])
period.rename(columns={'ID': 'PERIOD_UUID'}, inplace=True)
proposed_periods = hl_sched_prop_period.merge(period, on='PERIOD_UUID', how='left')
proposed_periods = proposed_periods.drop(columns=['PERIOD_UUID'])
# Merging schedule phases and periods
hl_schedulable_phase = hl_schedulable_phase.merge(phase, on='PHASE_UUID', how='left')
hl_schedulable_phase = hl_schedulable_phase.drop(columns=['PHASE_UUID'])
schedule_phases_periods = hl_schedulable_phase.merge(proposed_periods, on='SCHEDULABLE_PHASE_UUID', how='left')
grouped_schedule_phases_periods = schedule_phases_periods.groupby('SCHEDULABLE_PHASE_UUID').agg({
    'ACTIVITY_UUID': 'first',
    'PHASE_AMOUNT': 'first',
    'PHASE_DURATION': 'first',
    'NAME': lambda x: x.tolist(),
    'PHASE_NAME': 'first'})   
grouped_schedule_phases_periods = grouped_schedule_phases_periods.reset_index()
grouped_schedule_phases_periods = grouped_schedule_phases_periods.drop(columns=['SCHEDULABLE_PHASE_UUID'])

grouped_merged_schedule_phase_phase = grouped_schedule_phases_periods.groupby('ACTIVITY_UUID').agg(lambda x: x.tolist()) # grouping information by activities
grouped_merged_schedule_phase_phase = grouped_merged_schedule_phase_phase.reset_index()
grouped_merged_schedule_phase_phase['NEW_AMOUNT'] = grouped_merged_schedule_phase_phase['PHASE_AMOUNT'].copy()

for i, (dur, amt) in enumerate(zip(grouped_merged_schedule_phase_phase['PHASE_DURATION'], grouped_merged_schedule_phase_phase['PHASE_AMOUNT'])):
    for j, (d, a) in enumerate(zip(dur, amt)):
        # If the duration is 'months', multiply the amount by 4
        if d == 'MONTHS':
            grouped_merged_schedule_phase_phase.at[i, 'NEW_AMOUNT'][j] = a * 4
grouped_merged_schedule_phase_phase = grouped_merged_schedule_phase_phase.drop(columns=['PHASE_AMOUNT', 'PHASE_DURATION'])
grouped_merged_schedule_phase_phase = grouped_merged_schedule_phase_phase.rename(columns={'NEW_AMOUNT': 'PHASE_AMOUNT'})

# Q1 forming
# Merge activiy and activity list tables
merged_activity_list_activity = activity_list.merge(activity, on='ID', how='left')
merged_activity_list_activity = merged_activity_list_activity.drop(columns=['ACTIVITY_ID']) # drop activity ID column
merged_activity_list_activity.rename(columns={'ID': 'ACTIVITY_UUID'}, inplace=True)
# Merge total activity and schedule info
q1 = merged_activity_list_activity.merge(grouped_merged_schedule_phase_phase, on='ACTIVITY_UUID', how='left')

# Final treatment (bad practice)(last resort)
def rearrange(arr):
    if arr[0] == "Preparation" and arr[1] == "Commissioning":
        return [arr[0], arr[2], arr[1]]
    elif arr[0] == "Installation" and arr[1] == "Preparation":
        return [arr[1], arr[0], arr[2]]
    elif arr[0] == "Installation" and arr[1] == "Commissioning":
        return [arr[2], arr[0], arr[1]]
    elif arr[0] == "Commissioning" and arr[1] == "Preparation":
        return [arr[1], arr[2], arr[0]]
    elif arr[0] == "Commissioning" and arr[1] == "Installation":
        return [arr[2], arr[1], arr[0]]
    else:
        return arr

q1['PHASE_NAME_1'] = q1['PHASE_NAME'].apply(rearrange)
q1['PHASE_AMOUNT_1'] = q1.apply(lambda row: [row['PHASE_AMOUNT'][row['PHASE_NAME'].index(val)] for val in row['PHASE_NAME_1']], axis=1)
q1['NAME_1'] = q1.apply(lambda row: [row['NAME'][row['PHASE_NAME'].index(val)] for val in row['PHASE_NAME_1']], axis=1)
q1 = q1.drop(columns=['PHASE_NAME', 'PHASE_AMOUNT', 'NAME', 'PHASE_NAME_1'])
q1 = q1.rename(columns={'PHASE_NAME_1': 'PHASE_NAME', 'PHASE_AMOUNT_1': 'PHASE_AMOUNT', 'NAME_1': 'PERIOD'})
q1 = q1.drop(columns=['ACTIVITY_VERSION'])

print("\n", q1.head(10))
print("Shape: ", q1.shape)
#print(q1.nunique())

q1.to_csv('..\data\processed\q1.csv', index=False)
