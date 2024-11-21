Emotion Detection from Text: Machine Learning Project
==============================

## Overview

This project is a complete end-to-end machine learning pipeline to detect emotions in text provided by users. The system utilizes Natural Language Processing (NLP) and machine learning to classify emotions (such as happy, sad, angry, etc.) from user-provided text. The pipeline is fully automated, from model training to deployment, using CI/CD, Docker, and AWS EC2.

## Key Features:
- **Data Management**: Efficient data processing, versioning, and transformation using DVC.
- **Model Training & Evaluation**: Tracks model performance with DVC and Dagshub for versioning and model management.
- **Deployment**: The app is containerized using Docker and deployed to AWS EC2 for scalable hosting.
- **Automation**: Complete pipeline automation using GitHub Actions, ensuring smooth continuous integration and deployment.





## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


## Key Steps in the Workflow

### 1. Project Initialization
- **Cookiecutter Template**: Scaffolded the project using Cookiecutter to ensure a clean, maintainable, and standardized structure.
- **Logging and Exception Handling**: Created custom logging and exception handling files to manage errors efficiently throughout the pipeline.

### 2. Data Management with DVC
- **DVC Initialization**: Initialized DVC (Data Version Control) to manage data, track changes, and ensure reproducibility.
- **Pipeline Stages**: Created a DVC pipeline that includes the following stages:
  - **Data Injection**: Inject raw data into the system.
  - **Data Transformation**: Clean and transform the data.
  - **Feature Engineering**: Extract relevant features for model training.
  - **Model Building**: Train and evaluate machine learning models.
  - **Model Evaluation**: Evaluate model performance metrics.
  - **Model Registration**: Register the best model for production use.
  - **Stage Transaction**: Transition the model from staging to production.

### 3. Model Development
- Developed a baseline machine learning model to classify emotions in text.
- **Model Evaluation**: Regular performance tracking using Dagshub as the MLflow tracking URI.
- Implemented model tests to ensure correct model loading and inference behavior.

### 4. Flask API for Real-Time Inference
- Built a Flask web application to serve the emotion detection model via a RESTful API for real-time inference.

### 5. Model Registration & Staging
- Created separate scripts for model registration and model staging that move the best-performing model from staging to production.

### 6. CI/CD Pipeline with GitHub Actions
- **Linting**: Linted all scripts using flake8 to maintain code quality.
- **Model Testing**: Automated model tests with unittest.
- **DVC Pipeline Execution**: Ran the DVC pipeline using `dvc repro`.
- **Model Promotion**: Automatically promoted the model to production after successful tests.
- **Docker Build and Push**: Automated the Docker image build process and pushed the image to both Docker Hub and AWS ECR.
- **Deployment**: Automated the deployment of the Docker container to AWS EC2 instances.

### 7. Containerization and Deployment
- **Dockerization**: Containerized the Flask app along with the trained model into a Docker image.
- **Deployment to EC2**: Deployed the Docker container to an AWS EC2 instance for scalable production hosting.
- Docker images are pushed to both Docker Hub and AWS ECR for reliable version control and deployment.
- Automated deployment from both Docker Hub and AWS ECR using GitHub Actions.

### 8. GitHub Secrets
- Managed sensitive variables and credentials securely using GitHub Secrets, including AWS credentials, Docker Hub credentials, and Dagshub access tokens.

## GitHub Actions Workflow

The CI/CD pipeline is defined using GitHub Actions, which automates all critical tasks such as testing, Docker image building, and deployment. The pipeline is divided into the following jobs:

1. **Project Testing**:
   - Lints the code.
   - Runs unit tests.
   - Executes the DVC pipeline to ensure data and model integrity.
  
2. **Docker Build & Push to AWS ECR**:
   - Logs into AWS ECR.
   - Builds the Docker image.
   - Tags and pushes the image to AWS ECR for deployment.

3. **AWS EC2 Deployment**:
   - Logs into AWS EC2 via SSH.
   - Pulls the latest Docker image from AWS ECR.
   - Deploys the container on AWS EC2.

## Secrets Configuration

To securely manage environment variables, the following secrets are required in your GitHub repository:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_ECR_REGISTRY`
- `AWS_REGION`
- `DAGSHUB_PAT`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `DOCKER_IMAGE_NAME`
- `EC2_DEPLOYMENT_HOST`
- `EC2_DEPLOYMENT_SSH_KEY`
- `EC2_HOST`
- `EC2_SSH_KEY`
- `EC2_USER`

## How to Use

### Step 1: Install Dependencies
To get started with the project, install the required Python dependencies:

```bash
pip install -r requirements.txt
