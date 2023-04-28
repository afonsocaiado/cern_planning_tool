import pandas as pd
import numpy as np

import gower
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_nearest_neighbours(new_activity_json, predict_nans, k=10, knnw=0.3, nlpw=0.7):

    activities = pd.read_csv('..\..\data\\tables\\activity.csv', encoding='unicode_escape')

    relevant_columns = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']
    activities_relevant = activities.loc[:, relevant_columns]

    # Convert the JSON object to a pandas DataFrame
    new_activity = pd.DataFrame.from_dict(new_activity_json, orient='index').T
    new_activity_relevant = new_activity[relevant_columns]

    # Calculate text embeddings for the 'TITLE' field, if it's not np.nan
    if pd.isna(new_activity_json['TITLE']):
        title_similarities = None
    else:
        vectorizer = TfidfVectorizer(stop_words='english')
        title_embeddings = vectorizer.fit_transform(activities['TITLE'])
        new_title_embedding = vectorizer.transform([new_activity_json['TITLE']])
        title_similarities = cosine_similarity(new_title_embedding, title_embeddings).flatten()

    nan_columns = []

    if predict_nans:
        # Remove columns with NaN values
        for col in new_activity_relevant.columns:
            if new_activity_relevant[col].isnull().any():
                nan_columns.append(col)

        new_entry_not_nan = new_activity_relevant.drop(columns=nan_columns)
        activities_relevant_filtered = activities_relevant.drop(columns=nan_columns)

        if activities_relevant_filtered.shape[1] > 0:
            # Calculate the Gower distance matrix
            distance_matrix = gower.gower_matrix(activities_relevant_filtered, new_entry_not_nan)
        else:
            distance_matrix = np.zeros((activities.shape[0], 1))
    else:
        # Calculate the Gower distance matrix
        distance_matrix = gower.gower_matrix(activities_relevant, new_activity_relevant)

    if title_similarities is not None and len(nan_columns) != len(relevant_columns):
        # Combine Gower distance and cosine similarity (scaled to [0, 1] range)
        combined_matrix = (1 - title_similarities.reshape(-1, 1)) * nlpw + distance_matrix * knnw
    elif title_similarities is not None:
        # If all categorical columns are np.nan, use cosine similarity alone
        combined_matrix = 1 - title_similarities.reshape(-1, 1)
    else:
        # If 'TITLE' is np.nan, use Gower distance alone
        combined_matrix = distance_matrix

    # Find the indices of the k nearest neighbors
    knn_indices = combined_matrix.flatten().argsort()[:k]
    # Add the similarity score to the activity
    knn_scores = np.clip(combined_matrix.flatten()[knn_indices], 0, 1) # Restrict the score to stay within the range
    similarity_scores = 1 - knn_scores # Convert the score from a dissimilarity score to a similarity score
    # Get the k nearest neighbors from the original DataFrame
    knn = activities.iloc[knn_indices].copy()
    knn.loc[:, 'similarity_score'] = similarity_scores

    knn['CREATION_DATE'] = pd.to_datetime(knn['CREATION_DATE'], format='%d-%b-%y')
    knn = knn.sort_values(by=['similarity_score', 'CREATION_DATE'], ascending=[False, False])

    return knn, nan_columns

def make_suggestions_knn(knn, columns_to_suggest, print_suggestions=False):

    suggestions = {}

    for col in columns_to_suggest:
        suggestions[col] = []

    for col in suggestions.keys():
        max_list_size = 5
        col_values = knn[col].dropna().value_counts().index.tolist()[:max_list_size]
        suggestions[col] = col_values

    if print_suggestions:
        print("Suggested values based on the most common values within the nearest neighbours:")
        for col, values in suggestions.items():
            print(f"\nSuggestions for '{col}':")
            for index, value in enumerate(values):
                print(f"{index + 1}. {value} ({knn[col].value_counts()[value]} occurrences)")

    return suggestions
