# Plan Prediction Tool API

This API is designed to generate similar activities, activity suggestions, and contribution suggestions based on a given activity's details and confirmed contributions.

## Table of Contents

- [Requirements](#requirements)
- [API Structure](#api-structure)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [API Maintenance](#api-maintenance)
- [Running Tests](#running-tests)
- [Notes](#notes)

## Requirements

- Python 3.10
- Docker and Docker Compose

## API Structure

The structure of the present API can be found in [API Structure](./wiki/api_structure.md).

## Setup

1. Make sure you have Docker and Docker Compose installed on your system.
2. Clone this repository.
3. Navigate to the api folder (current location).

## Running the Application

1. Open a terminal and run `docker-compose up --build`. This will build the Docker image (if needed) and start the API.
2. Access the API at `http://localhost:5000/`. If you want to access the API from another machine on the same network, you can use the IP address of the machine running the API instead of "localhost". For example, if the IP address of the machine running the API is 192.168.1.100, you can access the API at http://192.168.1.100:5000/.

## API Endpoints

This API provides the following endpoints:

- `/get_similar_activities`
- `/get_activity_suggestions`
- `/get_initial_contribution_suggestions`
- `/get_combined_contribution_suggestions`

For detailed information on the request and response formats for each endpoint, please refer to the [API Endpoints Documentation](./wiki/api_endpoints.md).

## API Maintenance

More information and documentation on how to maintain and update the api can be found in the wiki folder. 

- [How to keep the datasets up-to-date](./wiki/update_data.md)
- [How to incorporate a new field into the model](./wiki/incorporate_new_field.md)
- [How to modify the returned activity fields](./wiki/change_returned_fields.md)
- [How to modify the datasets](./wiki/modify_data.md)

## Running Tests

To run the tests, execute the following command in your terminal:

```console
python test_api.py
```

## Future Work

All future work and improvements for the present work can be found [here](wiki/future_work.md).
