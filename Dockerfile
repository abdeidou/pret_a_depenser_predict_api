# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for the data folder
RUN mkdir /app/data

# Copy the contents of the data folder into the container at /app/data
COPY data /app/data

# Copy the rest of your application code into the container at /app
COPY . /app/

# Define environment variable
ENV FLASK_APP app.py

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
