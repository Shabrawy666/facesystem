# Use an official slim Python image
FROM python:3.11-slim
RUN apt-get update && apt-get install -y zstd

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies for psycopg2 and opencv-python
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libpq-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire project
COPY . .

# Ensure your image/stored_image folders exist
RUN mkdir -p /app/stored_images /app/temp_images

# Expose the port your app runs on (Gunicorn default: 8000, Fly.io default: 8080, Flask dev server default: 5000)
EXPOSE 8080

# Run with Gunicorn for production
# (change app:app to the module where your Flask instance lives---common: app:app, run:app, or wsgi:app)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "run:app"]

# Alternative (if your main Flask instance is in app.py):
# CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]