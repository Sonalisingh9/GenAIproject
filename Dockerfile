# Use a lightweight official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8080

# Set the working directory in the container
WORKDIR /app

# Copy the dependency file and install dependencies
COPY requirements.txt .
# Use --no-cache-dir for smaller image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
# Ensure your local structure matches this: app/main.py -> /app/app/main.py
COPY app/ /app/app

# Command to run the application with Gunicorn/Uvicorn workers (Recommended for production)
# This command starts the production server
CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]