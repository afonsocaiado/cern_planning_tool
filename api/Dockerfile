# Use the official Python image as the base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Add this line after your existing COPY or ADD commands
COPY entrypoint.sh /app/

# Replace your existing CMD or ENTRYPOINT with this line
ENTRYPOINT ["/app/entrypoint.sh"]

# Expose the port the app will run on
EXPOSE 5000