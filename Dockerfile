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

# Install production dependencies.
RUN pip install Flask gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app