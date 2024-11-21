# Stage 1: Install dependencies in a temporary build image
FROM python:3.10-slim AS builder

WORKDIR /app
# Copy requirements.txt and setup.py into the image
COPY flask_app/requirements.txt ./requirements.txt
COPY setup.py ./setup.py
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image, copy only the necessary files
FROM python:3.10-slim

WORKDIR /app

# Copy app files
COPY src ./src
COPY models ./models/
COPY flask_app ./flask_app
COPY setup.py ./setup.py

# Copy the installed dependencies from the builder image
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Check if gunicorn is installed, and its location
RUN if ! pip show gunicorn; then echo "Gunicorn is NOT installed"; else echo "Gunicorn is installed"; fi
RUN which gunicorn || echo "Gunicorn not found in PATH"

# Expose the application port
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
