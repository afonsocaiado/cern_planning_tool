# Update Activity and Contributions datasets

This document explains how to update the activity and contributions datasets to ensure that the models are working on the latest data.

### Step 1: Fetch data from PLAN databases

Manually replace all csv files located in the [original_tables](../data/csv_generation/original_tables) with the latest version of the PLAN databases.

### Step 2: Go to the desired script

1. Navigate to the [data](../data) folder. This is where you'll find everything related to the data.
2. Enter the [csv_generation](../data/csv_generation) folder. This is where the generation and updating of the final datasets is handled.

### Step 3: Run the csv updating python script

Run the file [generate_csv](../data/csv_generation/generate_csv.py). Both csvs should now be up-to-date and ready for use by the algorithms.

### Future work

- Ideally, we should figure out a way to automatically to this process. An API endpoint that runs periodically should do the job. Main thing that held this back was to automate Step 1, because of database access.