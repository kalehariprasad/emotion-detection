FROM python:3.10.15

# Set the working directory
WORKDIR /app

# Copy the source files
COPY src/custom_logging/__init__.py ./custom_logging/
COPY src/exeption/__init__.py ./exception/
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 5000

# Command to run the application
CMD ["python", "flask_app/app.py"]
