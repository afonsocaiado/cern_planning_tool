import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import joblib
import utils
import warnings

warnings.filterwarnings("ignore")

# Load the saved model
model = joblib.load('models/clustering_model.joblib')

def suggest_values(activity):

    # Create a DataFrame with the new activity
    new_data = pd.DataFrame([activity], columns=activity.keys())

    # Identify the missing fields
    missing_fields = new_data.columns[new_data.isna().any()].tolist()

    # If all fields are present, return the activity as is
    if not missing_fields:
        return activity

    # Scale the features
    # new_data = model.transform(new_data)
    new_data = utils.remove_nans(new_data)


    for col in new_data.columns:
        encoder = joblib.load("models/" + col + ".joblib")
        new_data[col] = encoder.transform(new_data[col])

    scaler = joblib.load("models/" + col + ".joblib")

    print(new_data)

    new_data = new_data.values.flatten()

    new_data = scaler.transform(new_data)

    print(new_data.head())

    # Cluster the new activity
    new_cluster = model.predict(new_data)[0]

    print(new_cluster)

    # Get the closest cluster in the dataset
    cluster_distances = pd.DataFrame({
        'cluster': model.labels_,
        'distance': model.transform(new_data)[:, new_cluster]
    })
    closest_cluster = cluster_distances.loc[cluster_distances['distance'].idxmin(), 'cluster']

    # Retrieve the data points in the closest cluster
    cluster_data = model['data'][model.labels_ == closest_cluster]

    # Calculate the means of the features for the cluster data
    cluster_means = cluster_data.mean()

    # Suggest values for missing fields based on the cluster means
    suggestions = {}
    for field in missing_fields:
        if pd.isna(activity[field]):
            suggestions[field] = cluster_means[field]

    # Update the activity with the suggested values
    activity.update(suggestions)

    return activity



    

activity = {
    'GROUP_RESPONSIBLE_NAME': 'SY-STI',
    'RESPONSIBLE_WITH_DETAILS': 'LIONEL HERBLIN (TE-CRG-OP)',
    'ACTIVITY_TYPE_EN': 'Consolidation & upgrade/Other',
    'WBS_NODE_CODE': 'NONE',
    'FACILITY_NAMES': 'LHC Machine',
    'PREPARATION_DURATION': float("nan"),
    'INSTALLATION_DURATION': 2,
    'COMMISSIONING_DURATION': 3
}

suggested_activity = suggest_values(activity)
print(suggested_activity)