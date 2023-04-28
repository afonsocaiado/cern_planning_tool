import pandas as pd
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gower

def get_nearest_neighbours(new_activity_json, k, predict_nans):

    activities = pd.read_csv('..\..\data\\tables\\activity.csv', encoding='unicode_escape')

    relevant_columns = ['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']
    activities_relevant = activities.loc[:, relevant_columns]

    # Convert the JSON object to a pandas DataFrame
    new_activity = pd.DataFrame.from_dict(new_activity_json, orient='index').T
    new_activity_relevant = new_activity[relevant_columns]

    # k = 30

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
        combined_matrix = (1 - title_similarities.reshape(-1, 1)) * 0.5 + distance_matrix * 0.5
    elif title_similarities is not None:
        # If all categorical columns are np.nan, use cosine similarity alone
        combined_matrix = 1 - title_similarities.reshape(-1, 1)
    else:
        # If 'TITLE' is np.nan, use Gower distance alone
        combined_matrix = distance_matrix

    # Find the indices of the k nearest neighbors
    knn_indices = combined_matrix.flatten().argsort()[:k]
    # Get the k nearest neighbors from the original DataFrame
    knn = activities.iloc[knn_indices]

    return knn, nan_columns

def contributions_knn_phase(knn, phase_name):

    contributions_df = pd.read_csv('..\..\data\\tables\\contributions.csv', encoding='unicode_escape')

    # Filter contributions by the given phase
    contributions_in_phase = contributions_df[contributions_df['PHASE_NAME'] == phase_name]
    merged_df = pd.merge(knn, contributions_in_phase, on='ACTIVITY_UUID', how='inner')

    common_contributions = merged_df.groupby(['CONTRIBUTION_TYPE', 'ORG_UNIT_CODE']).size().reset_index(name='count').sort_values(by='count', ascending=False).head(5)
    
    # Calculate the confidence percentage for each contribution type and organization unit
    total_contributions = merged_df.shape[0]
    common_contributions['confidence'] = common_contributions['count'] / total_contributions * 100
    
    common_contributions = common_contributions.drop(columns=['count'])

    return common_contributions

def contributions_knn_phase_varying(knn, phase_name, threshold=0.8):

    contributions_df = pd.read_csv('..\..\data\\tables\\contributions.csv', encoding='unicode_escape')

    # Filter contributions by the given phase
    contributions_in_phase = contributions_df[contributions_df['PHASE_NAME'] == phase_name]
    merged_df = pd.merge(knn, contributions_in_phase, on='ACTIVITY_UUID', how='inner')

    common_contributions = merged_df.groupby(['CONTRIBUTION_TYPE', 'ORG_UNIT_CODE']).size().reset_index(name='count').sort_values(by='count', ascending=False)
    
    # Calculate the cumulative percentage of the count column
    total_count = common_contributions['count'].sum()
    common_contributions['cumulative_percentage'] = common_contributions['count'].cumsum() / total_count

    # # Filter the rows where the cumulative percentage is less than or equal to the threshold
    # filtered_contributions = common_contributions[common_contributions['cumulative_percentage'] <= threshold]

    filtered_contributions = filtered_contributions.drop(columns=['count', 'cumulative_percentage'])

    return filtered_contributions

def contributions_knn(knn):

    # contributions_df = pd.read_csv('..\..\data\\tables\\contributions.csv', encoding='unicode_escape')

    suggestions = {}

    for phase in ['Preparation', 'Installation', 'Commissioning']:
        
        suggestions[phase] = contributions_knn_phase(knn, phase)

    return suggestions

def generate_association_rules(activities_df, contributions_df, min_support=0.05, min_confidence=0.5):
    # Merge the activities and contributions dataframes on the ACTIVITY_UUID column
    merged_df = pd.merge(activities_df, contributions_df, on='ACTIVITY_UUID')

    # Remove rows with missing values in 'PHASE_NAME', 'ORG_UNIT_CODE', or 'CONTRIBUTION_TYPE' columns
    merged_df = merged_df.dropna(subset=['PHASE_NAME', 'ORG_UNIT_CODE', 'CONTRIBUTION_TYPE'])

    # # Create a list of lists, where each inner list contains the combined phase, UNIT_ORG_CODE, and CONTRIBUTION_TYPE for a single activity
    # merged_df['UNIT_CONTRIBUTION'] = merged_df['PHASE_NAME'] + '_' + merged_df['ORG_UNIT_CODE'] + '_' + merged_df['CONTRIBUTION_TYPE']
    # grouped_contributions = merged_df.groupby('ACTIVITY_UUID')['UNIT_CONTRIBUTION'].apply(list)

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

def combined_suggestions(new_activity_json, confirmed_contributions_json, k=10, max_list_size=5, min_support=0.03, min_confidence=0.5):
    
    # Load activities and contributions data from CSV files
    activities_df = pd.read_csv('..\..\data\\tables\\activity.csv', encoding='unicode_escape')
    contributions_df = pd.read_csv('..\..\data\\tables\contributions.csv', encoding='unicode_escape')

    # Convert the confirmed_contributions_json to a pandas DataFrame
    confirmed_contributions_df = pd.DataFrame(confirmed_contributions_json)

    # Find the k-nearest neighbors and their associated contributions
    knn, nan_columns = get_nearest_neighbours(new_activity_json, k, False)
    knn_suggestions = contributions_knn(knn)

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