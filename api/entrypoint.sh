#!/bin/sh

# Run the tests
echo "Running tests..."
python test_api.py

# Check if the tests were successful
if [ $? -eq 0 ]; then
  echo "Tests passed. Starting the application..."
  # Start your application (replace the following line with the command to start your application)
  python api.py
else
  echo "Tests failed. Aborting startup."
  exit 1
fi
