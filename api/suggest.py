import pandas as pd
import numpy as np
import gower
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to get k nearest neighbors for a new activity
def get_nearest_neighbours(new_activity_json, predict_nans, k=10):

    """Find the k nearest neighbors for a given activity, using the activity dataset.

    Args:
        new_activity_json (dict): The new activity as a JSON object, with the following keys: 'TITLE', 'GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', and 'CREATOR_NAME'.
        predict_nans (bool): A boolean flag indicating whether or not to calculate the neighbors based on the available (considering intended empty fields) information in the activity. True means we only use the non-missing data.
        k (int, optional): The number of neighbors to return. Defaults to 10.

    Returns:
        knn (pd.DataFrame): A pandas DataFrame containing the k nearest neighbors in the activity dataset.
        nan_columns (list): A list of column names that had missing information in the new activity (for later prediction purposes).
    """

    activities = pd.read_csv('./data/activity.csv', encoding='unicode_escape')

    relevant_columns = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME'] # Categorical variables used for similarity calculation
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
        combined_matrix = (1 - title_similarities.reshape(-1, 1)) * 0.7 + distance_matrix * 0.3
    elif title_similarities is not None:
        # If all categorical columns are np.nan, use cosine similarity alone
        combined_matrix = 1 - title_similarities.reshape(-1, 1)
    else:
        # If 'TITLE' is np.nan, use Gower distance alone
        combined_matrix = distance_matrix

    # Find the indices of the k nearest neighbors and their similarity scores
    knn_indices = combined_matrix.flatten().argsort()[:k]
    # Add the similarity score to the activity
    knn_scores = np.clip(combined_matrix.flatten()[knn_indices], 0, 1) # Restrict the score to stay within the range
    similarity_scores = 1 - knn_scores # Convert the score from a dissimilarity score to a similarity score
    # Get the k nearest neighbors from the original DataFrame
    knn = activities.iloc[knn_indices].copy()
    knn.loc[:, 'similarity_score'] = similarity_scores

    # Sort the knn DataFrame by the 'similarity_score' first, and then by the 'CREATION_DATE' in descending order (most recent first) (when same similarity score, latest activity is returned)
    knn['CREATION_DATE'] = pd.to_datetime(knn['CREATION_DATE'], format='%d-%b-%y')
    knn = knn.sort_values(by=['similarity_score', 'CREATION_DATE'], ascending=[False, False])

    return knn, nan_columns

# Function to make suggestions based on k nearest neighbors
def make_suggestions_knn(knn, columns_to_suggest, max_list_size=5, print_suggestions=False):
    """Generate suggestions for the given columns based on the k nearest neighbors.

    Args:
        knn (pd.DataFrame): A pandas DataFrame containing the previously obtained k nearest neighbors in the activity dataset.
        columns_to_suggest (list): A list of column names for which suggestions should be generated.
        print_suggestions (bool, optional): A flag indicating whether to print the suggestions to the console. Defaults to False.

    Returns:
        suggestions (dict): A dictionary containing suggested values for each column in columns_to_suggest, based on the most common values within the nearest neighbors.
    """

    suggestions = {}

    for col in columns_to_suggest:
        suggestions[col] = []

    for col in suggestions.keys():
        col_values = knn[col].dropna().value_counts()
        total_occurrences = col_values.sum()
        suggestions[col] = [{'value': value, 'confidence': col_values[value] / total_occurrences * 100} for value in col_values.index.tolist()[:max_list_size]]

    if print_suggestions:
        print("Suggested values based on the most common values within the nearest neighbours:")
        for col, values in suggestions.items():
            print(f"\nSuggestions for '{col}':")
            for index, value in enumerate(values):
                print(f"{index + 1}. {value} ({knn[col].value_counts()[value]} occurrences)")

    return suggestions

# Function to get contribution suggestions for a specific phase based on k nearest neighbors
def contributions_knn_phase(knn, contributions_df, phase_name, max_list_size):
    """Generate common contribution requests (Type and Unit Code) for a given phase based on the k nearest neighbors.

    Args:
        knn (pd.DataFrame): A pandas DataFrame containing the previously obtained k nearest neighbors in the activity dataset.
        contributions_df (pd.DataFrame): A pandas DataFrame containing the contribution data.
        phase_name (str): The name of the phase for which to generate contribution suggestions ('Preparation', 'Installation', or 'Commissioning').

    Returns:
        common_contributions (pd.DataFrame): A pandas DataFrame containing the top 5 most common contribution types and their associated organization units, sorted by occurrence count in descending order.
    """

    # Filter contributions by the given phase
    contributions_in_phase = contributions_df[contributions_df['PHASE_NAME'] == phase_name]
    merged_df = pd.merge(knn, contributions_in_phase, on='ACTIVITY_UUID', how='inner')

    common_contributions = merged_df.groupby(['CONTRIBUTION_TYPE', 'ORG_UNIT_CODE']).size().reset_index(name='count').sort_values(by='count', ascending=False).head(max_list_size)
    
    # Calculate the confidence percentage for each contribution type and organization unit
    total_contributions = merged_df.shape[0]
    common_contributions['confidence'] = common_contributions['count'] / total_contributions * 100

    # Remove the 'count' column
    common_contributions = common_contributions.drop(columns=['count'])

    return common_contributions

# Function to get contribution suggestions based on k nearest neighbors
def contributions_knn(knn, max_list_size=5):
    """Generate contribution suggestions for each phase based on the k nearest neighbors.

    Args:
        knn (pd.DataFrame): A pandas DataFrame containing the previously obtained k nearest neighbors in the activity dataset.

    Returns:
        suggestions (dict): A dictionary containing suggested contributions for each phase ('Preparation', 'Installation', and 'Commissioning'). Each phase has a list of dictionaries with the top 5 most common contribution types and their associated organization units, sorted by occurrence count in descending order.
    """


    contributions_df = pd.read_csv('./data/contributions.csv', encoding='unicode_escape')

    suggestions = {}

    for phase in ['Preparation', 'Installation', 'Commissioning']:
        
        suggestions[phase] = contributions_knn_phase(knn, contributions_df, phase, max_list_size).to_dict(orient='records')

    return suggestions

# Function to generate association rules for contributions
def generate_association_rules(activities_df, contributions_df, min_support=0.03, min_confidence=0.5):
    """
    Generate association rules for contribution types and organizational units based on the given activities and contributions DataFrames.

    Args:
        activities_df (pd.DataFrame): A pandas DataFrame containing activity data.
        contributions_df (pd.DataFrame): A pandas DataFrame containing contribution data.
        min_support (float, optional): The minimum support threshold for the Apriori algorithm. Defaults to 0.05.
        min_confidence (float, optional): The minimum confidence threshold for the association rules. Defaults to 0.5.

    Returns:
        rules (pd.DataFrame): A DataFrame containing the generated association rules.
    """

    # Merge the activities and contributions dataframes on the ACTIVITY_UUID column
    merged_df = pd.merge(activities_df, contributions_df, on='ACTIVITY_UUID')

    # Remove rows with missing values in 'PHASE_NAME', 'ORG_UNIT_CODE', or 'CONTRIBUTION_TYPE' columns
    merged_df = merged_df.dropna(subset=['PHASE_NAME', 'ORG_UNIT_CODE', 'CONTRIBUTION_TYPE'])

    # Group by ACTIVITY_UUID and concatenate CONTRIBUTION_TYPE and ORG_UNIT_CODE as a tuple
    grouped_contributions = merged_df.groupby('ACTIVITY_UUID').apply(lambda x: [(row['PHASE_NAME'], row['CONTRIBUTION_TYPE'], row['ORG_UNIT_CODE']) for _, row in x.iterrows()])

    # Use the TransactionEncoder to transform the list of lists into a one-hot encoded DataFrame
    te = TransactionEncoder()
    te_array = te.fit(grouped_contributions).transform(grouped_contributions)
    one_hot_encoded_contributions = pd.DataFrame(te_array, columns=te.columns_)

    # Apply the Apriori algorithm to the one-hot encoded DataFrame
    frequent_itemsets = apriori(one_hot_encoded_contributions, min_support=min_support, use_colnames=True)

    # Generate association rules based on the frequent itemsets
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)

    return rules

# Function to get contribution suggestions based on k nearest neighbors and entered contributions
def combined_suggestions(new_activity_json, confirmed_contributions_json, k=10, max_list_size=5, min_support=0.03, min_confidence=0.5):
    """
    Generate combined contribution suggestions for each phase based on k nearest neighbors and association rules.

    Args:
        new_activity_json (dict): A dictionary containing the new activity data.
        confirmed_contributions_json (dict): A dictionary containing the contributions the user already entered for the new activity.
        k (int, optional): The number of nearest neighbors to consider. Defaults to 10.
        max_list_size (int, optional): The maximum number of suggestions to return for each phase. Defaults to 5.
        min_support (float, optional): The minimum support threshold for the Apriori algorithm. Defaults to 0.03.
        min_confidence (float, optional): The minimum confidence threshold for the association rules. Defaults to 0.5.

    Returns:
        knn_suggestions (dict): A dictionary containing suggested contributions for each phase ('Preparation', 'Installation', and 'Commissioning'). Each phase has a DataFrame with the top suggested contribution types and their associated organization units, sorted by confidence.
    """
    
    # Load activities and contributions data from CSV files
    activities_df = pd.read_csv('./data/activity.csv', encoding='unicode_escape')
    contributions_df = pd.read_csv('./data/contributions.csv', encoding='unicode_escape')

    # Convert the confirmed_contributions_json to a pandas DataFrame
    confirmed_contributions_df = pd.DataFrame(confirmed_contributions_json)

    # Find the k-nearest neighbors and their associated contributions
    knn, nan_columns = get_nearest_neighbours(new_activity_json, False, k)
    knn_suggestions = contributions_knn(knn, 10)

    # Generate association rules
    association_rules = generate_association_rules(activities_df, contributions_df, min_support, min_confidence)
    single_association_rules = association_rules[association_rules['antecedents'].apply(lambda x: len(x) == 1) & association_rules['consequents'].apply(lambda x: len(x) == 1)]

    # Initialize new_suggestions DataFrame
    new_suggestions = pd.DataFrame(columns=['PHASE_NAME', 'CONTRIBUTION_TYPE', 'ORG_UNIT_CODE', 'confidence'])

    # Iterate through single association rules and create new suggestions
    for _, rule in single_association_rules.iterrows():
        antecedent = next(iter(rule['antecedents']))
        phase, antecedent_type, antecedent_org_unit = antecedent

        # Check if the antecedent is in the confirmed contributions
        is_antecedent_confirmed = confirmed_contributions_df[
            (confirmed_contributions_df['PHASE_NAME'] == phase) &
            (confirmed_contributions_df['CONTRIBUTION_TYPE'] == antecedent_type) &
            (confirmed_contributions_df['ORG_UNIT_CODE'] == antecedent_org_unit)
        ].any().any()

        if is_antecedent_confirmed:

            consequent = next(iter(rule['consequents']))
            phase, consequent_type, consequent_org_unit = consequent

            # Create a new suggestion
            new_suggestion = pd.Series({
                'PHASE_NAME': phase,
                'CONTRIBUTION_TYPE': consequent_type,
                'ORG_UNIT_CODE': consequent_org_unit,
                'confidence': rule['confidence']*100
            })

            # Append the new suggestion to the new_suggestions DataFrame
            new_suggestion_df = pd.DataFrame([new_suggestion])
            new_suggestions = pd.concat([new_suggestions, new_suggestion_df], ignore_index=True)

    # Sort suggestions by confidence
    sorted_suggestions = new_suggestions.sort_values(by='confidence')    

    # Merge sorted suggestions with knn_suggestions
    for _, suggestion in sorted_suggestions.iterrows():

        temp_df = pd.DataFrame(columns=['CONTRIBUTION_TYPE', 'ORG_UNIT_CODE', 'confidence'])
        temp_row = pd.Series({
            'CONTRIBUTION_TYPE': suggestion['CONTRIBUTION_TYPE'],
            'ORG_UNIT_CODE': suggestion['ORG_UNIT_CODE'],
            'confidence': suggestion['confidence']
        })
        
        temp_row_df = pd.DataFrame([temp_row])
        temp_df = pd.concat([temp_df, temp_row_df], ignore_index=True)

        knn_suggestions_df = pd.DataFrame(knn_suggestions[suggestion['PHASE_NAME']])

        # Concatenate temp_df and knn_suggestions_df
        knn_suggestions[suggestion['PHASE_NAME']] = pd.concat([temp_df, knn_suggestions_df], ignore_index=True)

    for phase in knn_suggestions:
        knn_suggestions_df = pd.DataFrame(knn_suggestions[phase])

        # Filter out confirmed contributions for the current phase
        confirmed_contributions_phase_df = confirmed_contributions_df[confirmed_contributions_df['PHASE_NAME'] == phase]
        merged_df = knn_suggestions_df.merge(confirmed_contributions_phase_df, on=['CONTRIBUTION_TYPE', 'ORG_UNIT_CODE'], how='left', indicator=True)
        knn_suggestions_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge', 'PHASE_NAME'])

        knn_suggestions[phase] = knn_suggestions_df.drop_duplicates(subset=['CONTRIBUTION_TYPE', 'ORG_UNIT_CODE']).reset_index(drop=True).head(max_list_size)

    return knn_suggestions
