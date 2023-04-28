from flask import Flask, request, jsonify, abort
import suggest
from handlers import register_error_handlers
from utils import validate_input, validate_combined_suggestions_input

app = Flask(__name__)

register_error_handlers(app)

# Endpoint to get similar activities based on the new JSON activity
@app.route('/get_similar_activities', methods=['POST'])
def get_similar_activities():
    data = request.json
    validate_input(data)

    # Get k from the request, or use the default value if not provided
    k = int(request.args.get('k', 5)) # Number of nearest neighbors to return, default is 5

    try:
        # Get the nearest neighbours to the activity with the available information
        knn, columns_to_suggest = suggest.get_nearest_neighbours(data, True, k) 

        filtered_knn = knn[['ACTIVITY_UUID', 'similarity_score']] # Select which fields are to be returned in the response

        similar_activities = filtered_knn.to_dict(orient='records')

        # Return the similar activities as a JSON object
        return jsonify(similar_activities)
    
    except Exception as e:
        print(e)
        abort(500)

# Endpoint to get suggestions for the missing activity fields based on the new JSON activity
@app.route('/get_activity_suggestions', methods=['POST'])
def get_activity_suggestions():
    data = request.json
    validate_input(data)

    # Get k and max_list_size from the request, or use default values if not provided
    k = int(request.args.get('k', 10)) # Number of nearest neighbors, default is 10
    max_list_size = int(request.args.get('max_list_size', 5)) # Max number of returned values for each missing field, default is 5

    try:
        # Get the nearest neighbours to the activity with the available information
        knn, columns_to_suggest = suggest.get_nearest_neighbours(data, True, k)
        # From the group of nearest neighbours, get the suggestions for each field
        suggestions = suggest.make_suggestions_knn(knn, columns_to_suggest, max_list_size)

        # Return the suggestions as a JSON object
        return jsonify(suggestions)
    
    except Exception as e:
        print(e)
        abort(500)

# Endpoint to get suggestions for the contribution requests to make based on the newly created JSON activity
@app.route('/get_initial_contribution_suggestions', methods=['POST'])
def get_initial_contribution_suggestions():
    data = request.json
    validate_input(data)

    # Get k and max_list_size from the request, or use default values if not provided
    k = int(request.args.get('k', 10)) # Number of nearest neighbors, default is 10
    max_list_size = int(request.args.get('max_list_size', 5)) # Max number of suggested contributions for each phase, default is 5

    try: 
        # Get the nearest neighbours to the activity with the available information
        knn, columns_to_suggest = suggest.get_nearest_neighbours(data, False, k)
        # From the group of nearest neighbours, get the contribution suggestions for the 3 phases
        suggestions = suggest.contributions_knn(knn, max_list_size)

        # Return the suggestions as a JSON object
        return jsonify(suggestions)
    
    except Exception as e:
        print(e)
        abort(500)

# Endpoint to get suggestions for the contribution requests to make based on the newly created JSON activity and the entered contributions
@app.route('/get_combined_contribution_suggestions', methods=['POST'])
def get_combined_contribution_suggestions():
    data = request.json
    validate_combined_suggestions_input(data)

    # Access the activity and confirmed_contributions from the request body
    activity = data['activity']
    confirmed_contributions = data['confirmed_contributions']

    # Get k and max_list_size from the request, or use default values if not provided
    k = int(request.args.get('k', 10)) # Number of nearest neighbors, default is 10
    max_list_size = int(request.args.get('max_list_size', 5)) # Max number of suggested contributions for each phase, default is 5

    try: 
        # Generate suggestions using the combined_suggestions function
        suggestions = suggest.combined_suggestions(activity, confirmed_contributions, k, max_list_size)

        # Return the suggestions as a JSON object
        return jsonify({phase: suggestions[phase].to_dict(orient='records') for phase in suggestions})
    
    except Exception as e:
        print(e)
        abort(500)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
