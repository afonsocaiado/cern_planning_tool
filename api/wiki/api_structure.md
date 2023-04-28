# API Structure

This document explains how the API is structured

- [data](../data): Folder containing everything data related.
    - [csv_generation](../data/csv_generation): Folder containing everything related to the generation of the 2 final datasets.
        - [original_tables](../data/csv_generation/original_tables): Folder containing every used table from the PLAN databases as a csv.
        - [generate_csv](../data/csv_generation/generate_csv.py): Script to generate the final 2 datasets.
    - [activity.csv](../data/activity.csv): Final activity dataset.
    - [contributions.csv](../data/contributions.csv): Final contributions dataset.
- [wiki](../wiki): Folder containing more API documentation.
- [api.py](../api.py): The main entry point of your Flask application, containing the route definitions.
- [docker-compose.yml](../docker-compose.yml): Configuration file for Docker Compose, which allows you to define and run multi-container Docker applications.
- [Dockerfile](../Dockerfile): Instructions for building the Docker image of your application.
- [entrypoint.sh](../entrypoint.sh): A shell script to execute the tests and start the application when running it with Docker.
- [handlers.py](../handlers.py): Contains the custom error handlers for your application.
- [requirements.txt](../requirements.txt): Lists the required Python packages for your application.
- [suggest.py](../suggest.py): Contains the core logic for generating suggestions based on the input data.
- [test_api.py](../test_api.py): Contains the unit tests for your API endpoints.
- [utils.py](../utils.py): Contains utility functions used across your application.