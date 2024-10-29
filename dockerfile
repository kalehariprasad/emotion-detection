FROM python:3.10-slim

# Set the working directory
WORKDIR /app
COPY src ./src
COPY src/utils/__init__.py ./utils/          
COPY flask_app ./flask_app                   
COPY requirements.txt ./                      
COPY setup.py ./                              

# Install dependencies
RUN pip install -r requirements.txt

# Expose the application port
EXPOSE 5000

# Command to run the application
CMD ["python", "flask_app/app.py"]
