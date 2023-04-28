# Modify returned fields in the similar activity suggestion

This document explains how to modify which activity fields are returned in the similar activity suggestion endpoint.

### Step 1: Locate filtering line

Locate the following line in the `/get_similar_activities` endpoint in `api.py`:

```python
filtered_knn = knn[['ACTIVITY_UUID', 'similarity_score']] # Select which fields are to be returned in the response
```

It should be located [here](https://aistools-prod.cern.ch/stash/projects/PLAN/repos/plan-prediction-tool/browse/api/api.py#23)

### Step 2: Add the new field

Add the new field to the list.