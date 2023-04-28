# Incorporate a New Field in the similarity calculation

This document explains how to incorporate a new field in the model.

## Case 1: Categorical or Numerical Field

### Step 1: Add the new field to the relevant_columns list

Locate the following line in the `get_nearest_neighbours` function in `suggest.py`:

```python
relevant_columns = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME'] # Categorical variables used for similarity calculation
```

It should be located [here](https://aistools-prod.cern.ch/stash/projects/PLAN/repos/plan-prediction-tool/browse/api/suggest.py#29)

Add the new field to the `relevant_columns` list.

### Step 2: Modify the Gower distance calculation

if the new field is categorical or numerical, Gower distance will handle it should handle it without further modification.

## Case 2: Text Field

### Step 1: Add the new field

Calculate the text embeddings and cosine similarity as done for the 'TITLE' field in the `get_nearest_neighbours` function.

### Step 2: Weight adjustment (if needed)

If needed, adjust the weight of the Gower distance and cosine similarity in the combined_matrix calculation.

## Common steps for both cases

### Step 3: Update input validation to include the new field

Update the input validation functions in utils.py to validate the new field in the input data.

### Step 4: Update the documentation

Remember to also update any relevant parts of the README and other documentation files to reflect the addition of the new field.

