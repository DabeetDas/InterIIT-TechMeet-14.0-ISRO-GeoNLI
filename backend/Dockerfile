# Use an official Python runtime as a parent image. 
# 3.10 is a stable choice for PyTorch/Ultralytics.
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=True

# --- UPDATED SECTION ---
# Install system dependencies required for OpenCV
# We replaced 'libgl1-mesa-glx' with 'libgl1' for Debian 12 support
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# -----------------------

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# (Optional) Pre-download YOLO weights if they aren't local.
# It is better to have 'yolov8n.pt' (or your specific model) in the folder 
# and COPY it, so the app doesn't try to download it on every startup.

# Expose the port Cloud Run expects (default 8080)
ENV PORT=8080

# Use Gunicorn as the production WSGI server
# "app:app" means "look in module 'app.py' for the object named 'app'"
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app