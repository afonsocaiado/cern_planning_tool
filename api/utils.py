from flask import abort

def validate_input(data):
    """Validates the input data for the API endpoints.

    Checks if all required activity keys are present in the input data. If not, aborts the request with a 400 Bad Request status code.

    Args:
        data (dict): The input data as a JSON object.

    Raises:
        BadRequest: If any of the required keys are missing from the input data.
    """

    # List of required keys
    required_keys = ['TITLE', 'GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']

    # Check if the input data is empty
    if not data:
        abort(400, description="Bad Request: Empty Body")
        
    # Check for missing required keys in the input data
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        abort(400, description=f"Bad Request: Missing fields {missing_keys}")

def validate_combined_suggestions_input(data):
    """Validates the input data for the /get_combined_contribution_suggestions API endpoint.

    Checks if the required keys ('activity' and 'confirmed_contributions') are present in the input data, and if the required fields are present in the activity and confirmed_contributions data. If not, aborts the request with a 400 Bad Request status code.

    Args:
        data (dict): The input data as a JSON object.

    Raises:
        BadRequest: If any of the required keys are missing from the input data, or if any of the required fields are missing from the activity or confirmed_contributions data..
    """

    required_keys = ['activity', 'confirmed_contributions']
    required_activity_keys = ['TITLE', 'GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']
    required_contributions_keys = ['PHASE_NAME', 'CONTRIBUTION_TYPE', 'ORG_UNIT_CODE']

    # Check if the input data is empty
    if not data:
        abort(400, description="Bad Request: Empty Body")

    # Check for missing required keys in the input data
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        abort(400, description=f"Bad Request: Missing fields {missing_keys}")

    # Check for missing required fields in the 'activity' data
    activity = data['activity']
    missing_activity_keys = [key for key in required_activity_keys if key not in activity]
    if missing_activity_keys:
        abort(400, description=f"Bad Request: Missing activity fields {missing_activity_keys}")

    # Check for missing required fields in the 'confirmed_contributions' data
    confirmed_contributions = data['confirmed_contributions']
    if not isinstance(confirmed_contributions, list):
        abort(400, description="Bad Request: confirmed_contributions is not a list")
    
    for contribution in confirmed_contributions:
        missing_contribution_keys = [key for key in required_contributions_keys if key not in contribution]
        if missing_contribution_keys:
            abort(400, description=f"Bad Request: Missing contribution fields {missing_contribution_keys}")

