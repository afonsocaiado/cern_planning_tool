# Modifying Columns and Fields of the Final Datasets

This document will show you how to modify the columns and fields of the final datasets produced by the [generate_csv](../data/csv_generation/generate_csv.py) script.

## Overview

The [generate_csv](../data/csv_generation/generate_csv.py) script reads raw CSV files from the [original_tables](../data/csv_generation/original_tables) folder and processes them to generate two final datasets:

1. `activity.csv`
2. `contributions.csv`

These final datasets are stored in the [data](../data/csv_generation/data) folder.

## How to Modify the Final Datasets

To modify the columns and fields of the final datasets, you need to edit the [generate_csv](../data/csv_generation/generate_csv.py) script. Here's a step-by-step guide:

### 1. Locate the relevant section in the script

Find the section that processes the specific dataset you want to modify:

- For the `activity.csv` dataset, look for the `get_data_activities()` function.
- For the `contributions.csv` dataset, look for the `get_data_contributions()` function.

### 2. Modify the DataFrame processing steps

Inside the respective function, you will see several DataFrame processing steps, properly commented, such as:

- Reading the raw CSV files
- Dropping unnecessary columns
- Filtering, merging, and aggregating the DataFrames

To modify the columns and fields, you can adjust these processing steps accordingly. For example, if you want to add or remove columns, you can update the `drop()` or `filter()` functions.

### 3. Update the final output (shouldn't be needed)

After modifying the processing steps, ensure that the final output is saved correctly:

- For the `activity.csv` dataset, update the line `activity_df.to_csv('..\\activity.csv', index=False)`.
- For the `contributions.csv` dataset, update the line `contribution_group_and_phase_and_type.to_csv('..\\contributions.csv', index=False)`.

## Example: Adding a Column to the `activity.csv` Dataset

Suppose you want to add the `STATUS` column to the `activity.csv` dataset. Follow these steps:

1. Locate the `get_data_activities()` function in the `generate_csv.py` script.
2. Find the line where the `STATUS` column is dropped: `activity_list = activity_list.drop(columns=['ACTIVITY_TYPE_FR', 'PRIORITY_FR', 'STATUS', ...])`.
3. Remove `'STATUS'` from the list of columns to drop.
4. The `STATUS` column will now be included in the final `activity.csv` dataset.

After making the changes, run the [generate_csv](../data/csv_generation/generate_csv.py) script to generate the updated final datasets with the new columns and fields.