# Use Python 3.9 slim as the base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install numpy directly
RUN pip install --no-cache-dir numpy

# Copy evaluation script
COPY evaluate.py .

# Set the entrypoint
ENTRYPOINT ["python", "evaluate.py"]