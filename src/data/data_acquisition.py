import pandas as pd
import numpy as np

def get_data_activities():

    activity_list = pd.read_csv('..\..\data\original\\activity_list.csv', encoding='unicode_escape')
    activity = pd.read_csv('..\..\data\original\\activity.csv', encoding='unicode_escape')
    phase = pd.read_csv('..\..\data\original\phase.csv', encoding='unicode_escape')
    hl_schedulable_phase = pd.read_csv('..\..\data\original\hl_schedulable_phase.csv', encoding='unicode_escape')
    period = pd.read_csv('..\..\data\original\period.csv', encoding='unicode_escape')
    hl_sched_prop_period = pd.read_csv('..\..\data\original\hl_sched_prop_period.csv', encoding='unicode_escape')
    hl_sched_acce_period = pd.read_csv('..\..\data\original\hl_sched_acce_period.csv', encoding='unicode_escape')

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

    # activity_df forming
    # Merge activiy and activity list tables
    merged_activity_list_activity = activity_list.merge(activity, on='ID', how='left')
    merged_activity_list_activity = merged_activity_list_activity.drop(columns=['ACTIVITY_ID']) # drop activity ID column
    merged_activity_list_activity.rename(columns={'ID': 'ACTIVITY_UUID'}, inplace=True)
    # Merge total activity and schedule info
    activity_df = merged_activity_list_activity.merge(grouped_merged_schedule_phase_phase, on='ACTIVITY_UUID', how='left')

    def rearrange(arr):
        activity_order = {"Preparation": 1, "Installation": 2, "Commissioning": 3}
        sorted_arr = sorted(arr, key=lambda x: activity_order[x])
        return sorted_arr

    activity_df['PHASE_NAME_1'] = activity_df['PHASE_NAME'].apply(rearrange)
    activity_df['PHASE_AMOUNT_1'] = activity_df.apply(lambda row: [row['PHASE_AMOUNT'][row['PHASE_NAME'].index(val)] for val in row['PHASE_NAME_1']], axis=1)
    activity_df['NAME_1'] = activity_df.apply(lambda row: [row['NAME'][row['PHASE_NAME'].index(val)] for val in row['PHASE_NAME_1']], axis=1)
    activity_df = activity_df.drop(columns=['PHASE_NAME', 'PHASE_AMOUNT', 'NAME', 'PHASE_NAME_1'])
    activity_df = activity_df.rename(columns={'PHASE_NAME_1': 'PHASE_NAME', 'PHASE_AMOUNT_1': 'PHASE_AMOUNT', 'NAME_1': 'PERIOD'})

    # Separate Phase Duration
    activity_df[['PREPARATION_DURATION', 'INSTALLATION_DURATION', 'COMMISSIONING_DURATION']] = activity_df['PHASE_AMOUNT'].apply(lambda x: pd.Series(x))
    activity_df = activity_df.drop(columns=['PHASE_AMOUNT'])

    activity_df = activity_df.drop(columns=['ACTIVITY_VERSION'])

    # Uncomment this line if needed
    activity_df = activity_df.drop(columns=['PERIOD', 'PREPARATION_DURATION', 'INSTALLATION_DURATION', 'COMMISSIONING_DURATION'])

    # activity_df.to_csv('..\..\\api\\data\\activity.csv', index=False)

    # add 2 rows for the encoder to know missing value label in all intended columns
    new_row1 = {'TITLE': 'New liqufier in P6 for helium management', 'GROUP_RESPONSIBLE_NAME': pd.NaT, 'RESPONSIBLE_WITH_DETAILS': 'LIONEL HERBLIN (TE-CRG-OP)', 'ACTIVITY_TYPE_EN': 'Consolidation & upgrade/Other', 'WBS_NODE_CODE': 'NONE', 'FACILITY_NAMES': 'LHC Machine', 'PRIORITY_EN': pd.NaT, 'CREATOR_NAME': 'SYSTEM ADMINISTRATORS', 'CREATION_DATE': '19-NOV-21', 'LOCATION_INFORMATION': pd.NaT, 'GOAL': 'Improve the management of the helium inventory during long shutdowns.', 'IMPACT_NOT_DONE': 'Reduced flexibility in helium management.', 'PERIOD': [['LS2'], ['LS2'], ['LS2']], 'PREPARATION_DURATION': np.nan, 'INSTALLATION_DURATION': np.nan, 'COMMISSIONING_DURATION': np.nan}
    new_row2 = {'TITLE': 'Short Circuit Tests campaign at the end of LS2', 'GROUP_RESPONSIBLE_NAME': 'TE-MPE', 'RESPONSIBLE_WITH_DETAILS': pd.NaT, 'ACTIVITY_TYPE_EN': 'Consolidation & upgrade/Other', 'WBS_NODE_CODE': 'NONE', 'FACILITY_NAMES': 'LHC Machine', 'PRIORITY_EN': pd.NaT, 'CREATOR_NAME': 'SYSTEM ADMINISTRATORS', 'CREATION_DATE': '19-NOV-21', 'LOCATION_INFORMATION': 'The tests will be performed in all UAs and RRs, with limitation of access in front of the DFBs only on the day of the test.', 'GOAL': 'The objective of the Short Circuit Test campaign is threefold: validate the replacement of the water-cooled cable hoses in several points; validate the maintenance on the 13 kA energy extraction systems in all points put in operation and validate a permanent monitoring system for the 13 kA conical joints.', 'IMPACT_NOT_DONE': 'The bad cooling of the water-cooled cables could have catastrophic consequences on the integrity of the cables, while a bad conical joint could lead to local melting of the joint itself. In both cases, the stop could be of several weeks, if discovered during operation. The missing validation of the maintained EE systems could have important consequences on the integrity of the 13 kA circuits, with major consequences.', 'PERIOD': [['LS2'], ['LS2'], ['LS2']], 'PREPARATION_DURATION': 22, 'INSTALLATION_DURATION': 22, 'COMMISSIONING_DURATION': 22}
  
    activity_df.loc[len(activity_df)] = new_row1
    activity_df.loc[len(activity_df)] = new_row2

    activity_df.to_csv('..\..\data\\tables\\activity.csv', index=False)

def get_data_contributions():

    contribution_config_ref = pd.read_csv('..\..\data\original\\contribution_config_ref.csv', encoding='unicode_escape')
    contribution_field = pd.read_csv('..\..\data\original\\contribution_field.csv', encoding='unicode_escape')
    contribution_group = pd.read_csv('..\..\data\original\\contribution_group.csv', encoding='unicode_escape')
    contribution_group_period = pd.read_csv('..\..\data\original\\contribution_group_period.csv', encoding='unicode_escape')
    contribution_type = pd.read_csv('..\..\data\original\\contribution_type.csv', encoding='unicode_escape')
    period = pd.read_csv('..\..\data\original\period.csv', encoding='unicode_escape')
    phase = pd.read_csv('..\..\data\original\phase.csv', encoding='unicode_escape')
    org_unit = pd.read_csv('..\..\data\original\org_unit.csv', encoding='unicode_escape')
    org_unit_import = pd.read_csv('..\..\data\original\org_unit_import.csv', encoding='unicode_escape')
    
    # CONTRIBUTION CONFIG REF
    contribution_config_ref = contribution_config_ref.drop(columns=['PLAN_UUID', 'VERSION_UUID']) # based on uniques
    # print("\ncontribution_config_ref: \n", contribution_config_ref.nunique())

    # CONTRIBUTION FIELD
    contribution_field = contribution_field.drop(columns=['PLAN_UUID', 'VERSION_UUID', 'PRIMARY_FIELD_CODE']) # based on uniques
    # print("\ncontribution_field: \n", contribution_field.nunique())

    # CONTRIBUTION GROUP
    contribution_group = contribution_group.drop(columns=['UNCERTAINTY_LEVEL', 'GROUP_CONTRIBUTION_ID']) # based on uniques (0 uniques, same amount as UUID)
    contribution_group = contribution_group.drop(columns=['COMMENTS', 'CONTRIBUTOR_COMMENTS', 'FINISHED', 'VALIDATED', 'AVAILABLE', 'CUSTOM_PERIODS', 'PARENT_CONTRIBUTION_UUID']) # based on logic
    # contribution_group = contribution_group.query("FOR_COORDINATION == 0")
    # print("\ncontribution_group: \n", contribution_group.nunique())
    
    # CONTRIBUTION GROUP PERIOD
    # print("\ncontribution_group_period: \n", contribution_group_period.nunique())

    # CONTRIBUTION TYPE
    contribution_type = contribution_type.drop(columns=['PLAN_UUID', 'ACTIVE', 'CONTRIBUTION_TYPE_ID']) # based on uniques
    contribution_type = contribution_type.drop(columns=['NAME_FR', 'ORG_UNIT_UUID', 'ORDER_IN_TYPE', 'ORG_UNIT_CODE']) # based on logic  
    contribution_type.rename(columns={'ID': 'CONTRIBUTION_TYPE_UUID', 'NAME_EN': 'CONTRIBUTION_TYPE'}, inplace=True)
    # print("\ncontribution_type: \n", contribution_type.nunique())

    # PERIOD
    period_name = period.drop(columns=['PERIOD_ID', 'PLAN_UUID', 'START_DATE', 'END_DATE', 'ACTIVE', 'PARENT_UUID', 'PERIOD_ORDER'])
    period_name.rename(columns={'ID': 'PERIOD_UUID', 'NAME': 'PERIOD_NAME'}, inplace=True)
    # print("period_name: \n", period_name.nunique())

    # PHASE
    phase_name = phase.filter(items=['ID', 'NAME_EN'])
    phase_name.rename(columns={'ID': 'PHASE_UUID', 'NAME_EN': 'PHASE_NAME'}, inplace=True)
    # print("phase_name: \n", phase_name.nunique())

    # ORG UNIT
    org_unit = org_unit.drop(columns=['DESCRIPTION', 'PARENT', 'B_LEVEL', 'ORG_UNIT_VERSION']) # based on logic
    org_unit.rename(columns={'TYPE': 'ORG_UNIT_TYPE', 'CODE': 'ORG_UNIT_CODE'}, inplace=True)
    # print("\norg_unit: \n", org_unit.nunique())

    # ORG UNIT IMPORT
    org_unit_import = org_unit_import.drop(columns=['DESCRIPTION', 'PARENT']) # based on logic  
    # print("org_unit_import: \n", org_unit_import.nunique())

    # MERGES
    contribution_group_period_and_name = contribution_group_period.merge(period_name, on='PERIOD_UUID')
    contribution_group_period_and_name = contribution_group_period_and_name.drop(columns=['PERIOD_UUID'])
    # print("\ncontribution_group_period_and_name: \n", contribution_group_period_and_name.nunique())

    contribution_group_and_phase = contribution_group.merge(phase_name, on='PHASE_UUID', how='left')
    contribution_group_and_phase = contribution_group_and_phase.drop(columns=['PHASE_UUID'])

    # get the org unit name
    # Step 1: Create a new column in contribution_group_and_phase called 'UUID_to_fetch', containing the SECTION_UUID if it exists, otherwise the GROUP_UUID
    contribution_group_and_phase['UUID_to_fetch'] = contribution_group_and_phase.apply(
        lambda row: row['SECTION_UUID'] if pd.notna(row['SECTION_UUID']) else row['GROUP_UUID'], axis=1)
    # Step 2: Merge contribution_group_and_phase with org_unit using the 'UUID_to_fetch' column
    org_unit = org_unit.rename(columns={'ID': 'UUID_to_fetch'})  # Rename the ID column in org_unit for merging
    contribution_group_and_phase = contribution_group_and_phase.merge(org_unit, on='UUID_to_fetch', how='left')
    # Remove the 'UUID_to_fetch' column
    contribution_group_and_phase = contribution_group_and_phase.drop(columns=['UUID_to_fetch'])

    contribution_group_and_phase_and_type = contribution_group_and_phase.merge(contribution_type, on='CONTRIBUTION_TYPE_UUID', how='left')
    contribution_group_and_phase_and_type = contribution_group_and_phase_and_type.drop(columns=['CONTRIBUTION_TYPE_UUID'])
    # print("\ncontribution_group_and_phase_and_type: \n", contribution_group_and_phase_and_type.nunique())

    contribution_group_and_phase_and_type.to_csv('..\..\data\\tables\\contributions.csv', index=False)
    # contribution_group_and_phase_and_type.to_csv('..\..\\api\\data\\contributions.csv', index=False)

get_data_activities()
get_data_contributions()