# Stage 1: Install dependencies in a temporary build image
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt ./
COPY setup.py ./setup.py
# Install build dependencies
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

# Expose the application port
EXPOSE 5000

# Command to run the application
CMD ["python", "flask_app/app.py"]
