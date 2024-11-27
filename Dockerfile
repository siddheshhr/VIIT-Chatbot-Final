# Use an official Python runtime as a parent image
FROM python:3.9-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1

# Set work directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Install pip dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Install Hugging Face token as an environment variable (optional, replace with your method of secret management)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Health check to ensure the application is running
HEALTHCHECK CMD curl --fail http://localhost:8000/ || exit 1

# Use uvicorn to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]