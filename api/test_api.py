import unittest
from flask import json
from flask_testing import TestCase
from api import app

class TestAPI(TestCase):

    def create_app(self):
        app.config['TESTING'] = True
        return app

    def test_get_similar_activities_valid_input(self):
        """
        Test the /get_similar_activities endpoint with valid input data.
        This test case verifies that the endpoint returns a status code of 200 (OK)
        and that the response contains a list of similar activities when provided
        with a valid JSON activity object.
        """
        # Define a valid new_activity object with required fields
        new_activity = {
            "TITLE": "Example Title",
            "GROUP_RESPONSIBLE_NAME": "Group Name",
            "RESPONSIBLE_WITH_DETAILS": "Responsible Details",
            "ACTIVITY_TYPE_EN": "Activity Type",
            "WBS_NODE_CODE": "WBS Code",
            "CREATOR_NAME": "Creator Name"
        }
        # Send a POST request to the /get_similar_activities endpoint with the new_activity object
        response = self.client.post(
            '/get_similar_activities',
            data=json.dumps(new_activity),
            content_type='application/json'
        )
        # Check if the response has a status code of 200 (OK)
        self.assertEqual(response.status_code, 200)
        # Check if the response data is a list of similar activities
        self.assertTrue(isinstance(json.loads(response.data), list))

    def test_get_similar_activities_missing_field(self):
        """
        Test the /get_similar_activities endpoint with missing field in input data.
        This test case verifies that the endpoint returns a status code of 400 (Bad Request)
        when provided with a JSON activity object that is missing a required field.
        """
        # Define a new_activity object with a missing required field ("CREATOR_NAME")
        new_activity = {
            "TITLE": "Example Title",
            "GROUP_RESPONSIBLE_NAME": "Group Name",
            "RESPONSIBLE_WITH_DETAILS": "Responsible Details",
            "ACTIVITY_TYPE_EN": "Activity Type",
            "WBS_NODE_CODE": "WBS Code"
            # "CREATOR_NAME" is missing
        }
        # Send a POST request to the /get_similar_activities endpoint with the new_activity object
        response = self.client.post(
            '/get_similar_activities',
            data=json.dumps(new_activity),
            content_type='application/json'
        )
        # Check if the response has a status code of 400 (Bad Request)
        self.assertEqual(response.status_code, 400)

    def test_get_activity_suggestions_valid_input(self):
        """
        Test the /get_activity_suggestions endpoint with valid input data.
        This test case verifies that the endpoint returns a status code of 200 (OK)
        and a JSON object with suggestions when provided with a valid JSON activity object.
        """
        # Define a valid new_activity object
        new_activity = {
            "TITLE": "Example Title",
            "GROUP_RESPONSIBLE_NAME": "Group Name",
            "RESPONSIBLE_WITH_DETAILS": "Responsible Details",
            "ACTIVITY_TYPE_EN": "Activity Type",
            "WBS_NODE_CODE": "WBS Code",
            "CREATOR_NAME": "Creator Name"
        }
        # Send a POST request to the /get_activity_suggestions endpoint with the new_activity object
        response = self.client.post(
            '/get_activity_suggestions',
            data=json.dumps(new_activity),
            content_type='application/json'
        )
        # Check if the response has a status code of 200 (OK) and if the response data is a JSON object
        self.assertEqual(response.status_code, 200)
        self.assertTrue(isinstance(json.loads(response.data), dict))

    def test_get_activity_suggestions_missing_field(self):
        """
        Test the /get_activity_suggestions endpoint with missing field in input data.
        This test case verifies that the endpoint returns a status code of 400 (Bad Request)
        when provided with a JSON activity object that is missing a required field.
        """
        # Define a new_activity object with a missing required field ("CREATOR_NAME")
        new_activity = {
            "TITLE": "Example Title",
            "GROUP_RESPONSIBLE_NAME": "Group Name",
            "RESPONSIBLE_WITH_DETAILS": "Responsible Details",
            "ACTIVITY_TYPE_EN": "Activity Type",
            "WBS_NODE_CODE": "WBS Code"
            # "CREATOR_NAME" is missing
        }
        # Send a POST request to the /get_activity_suggestions endpoint with the new_activity object
        response = self.client.post(
            '/get_activity_suggestions',
            data=json.dumps(new_activity),
            content_type='application/json'
        )
        # Check if the response has a status code of 400 (Bad Request)
        self.assertEqual(response.status_code, 400)

    def test_get_initial_contribution_suggestions_valid_input(self):
        """
        Test the /get_contribution_suggestions endpoint with valid input.
        The response should have a 200 (OK) status code and return a JSON object.
        """
        # Prepare a valid new activity JSON object
        new_activity = {
            "TITLE": "Example Title",
            "GROUP_RESPONSIBLE_NAME": "Group Name",
            "RESPONSIBLE_WITH_DETAILS": "Responsible Details",
            "ACTIVITY_TYPE_EN": "Activity Type",
            "WBS_NODE_CODE": "WBS Code",
            "CREATOR_NAME": "Creator Name"
        }
        # Send the request to the endpoint
        response = self.client.post(
            '/get_initial_contribution_suggestions',
            data=json.dumps(new_activity),
            content_type='application/json'
        )
        # Check that the status code is 200 (OK) and the response is a JSON object
        self.assertEqual(response.status_code, 200)
        self.assertTrue(isinstance(json.loads(response.data), dict))

    def test_get_initial_contribution_suggestions_missing_field(self):
        """
        Test the /get_contribution_suggestions endpoint with a missing field in the input.
        The response should have a 400 (Bad Request) status code.
        """
        # Prepare a new activity JSON object with a missing field
        new_activity = {
            "TITLE": "Example Title",
            "GROUP_RESPONSIBLE_NAME": "Group Name",
            "RESPONSIBLE_WITH_DETAILS": "Responsible Details",
            "ACTIVITY_TYPE_EN": "Activity Type",
            "WBS_NODE_CODE": "WBS Code"
            # "CREATOR_NAME" is missing
        }
        # Send the request to the endpoint
        response = self.client.post(
            '/get_initial_contribution_suggestions',
            data=json.dumps(new_activity),
            content_type='application/json'
        )
        # Check that the status code is 400 (Bad Request)
        self.assertEqual(response.status_code, 400)

    def test_get_combined_contribution_suggestions_valid_input(self):
        """
        Test the /get_combined_contribution_suggestions endpoint with valid input.
        The response should have a 200 (OK) status code and return a JSON object.
        """
        # Prepare a valid request body with 'activity' and 'confirmed_contributions'
        data = {
            "activity": {
                "TITLE": "Example Title",
                "GROUP_RESPONSIBLE_NAME": "Group Name",
                "RESPONSIBLE_WITH_DETAILS": "Responsible Details",
                "ACTIVITY_TYPE_EN": "Activity Type",
                "WBS_NODE_CODE": "WBS Code",
                "CREATOR_NAME": "Creator Name"
            },
            "confirmed_contributions": [
                {
                    "PHASE_NAME": "Preparation",
                    "CONTRIBUTION_TYPE": "Type 1",
                    "ORG_UNIT_CODE": "Org Unit 1"
                }
            ]
        }
        # Send the request to the endpoint
        response = self.client.post(
            '/get_combined_contribution_suggestions',
            data=json.dumps(data),
            content_type='application/json'
        )
        # Check that the status code is 200 (OK) and the response is a JSON object
        self.assertEqual(response.status_code, 200)
        self.assertTrue(isinstance(json.loads(response.data), dict))

    def test_get_combined_contribution_suggestions_missing_field(self):
        """
        Test the /get_combined_contribution_suggestions endpoint with a missing field in the input.
        The response should have a 400 (Bad Request) status code.
        """
        # Prepare a request body with a missing field in the 'activity' object
        data = {
            "activity": {
                "TITLE": "Example Title",
                "GROUP_RESPONSIBLE_NAME": "Group Name",
                "RESPONSIBLE_WITH_DETAILS": "Responsible Details",
                "ACTIVITY_TYPE_EN": "Activity Type",
                "WBS_NODE_CODE": "WBS Code"
                # "CREATOR_NAME" is missing
            },
            "confirmed_contributions": [
                {
                    "PHASE_NAME": "Preparation",
                    "CONTRIBUTION_TYPE": "Type 1",
                    "ORG_UNIT_CODE": "Org Unit 1"
                }
            ]
        }
        # Send the request to the endpoint
        response = self.client.post(
            '/get_combined_contribution_suggestions',
            data=json.dumps(data),
            content_type='application/json'
        )
        # Check that the status code is 400 (Bad Request)
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
