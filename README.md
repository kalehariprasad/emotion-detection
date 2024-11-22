
## Emotion Detection from Text: Machine Learning Project

### Overview


This project is a complete **end-to-end machine learning pipeline** to detect emotions in text provided by users. The system utilizes **Natural Language Processing (NLP)** and machine learning to classify emotions (such as happy, sad, angry, etc.) from user-provided text. The pipeline is fully automated, from model training to deployment, **using CI/CD, Docker, and AWS EC2.**


### key Featurees :
•	**Data Management**: Efficient data processing, versioning, and transformation using DVC.

•	**Model Training & Evaluation**: Tracks model performance with DVC and Dagshub for versioning and model management.

•	**Deployment**: The app is containerized using Docker and deployed to AWS EC2 for scalable hosting.

•	**Automation**: Complete pipeline automation using GitHub Actions, ensuring smooth continuous integration and deployment.
 
## Key Steps in the Workflow

### 1.Project Initialization

•	**Cookiecutter Template**: Scaffolded the project using **Cookiecutter** to ensure a clean, maintainable, and standardized structure.

•	**Logging and Exception Handling**: Created custom logging and exception handling files to manage errors efficiently throughout the pipeline.

### 2.Data Management with DVC
•	**DVC Initialization**: Initialized DVC (Data Version Control) to manage data, track changes, and ensure reproducibility.

•	**Pipeline Stages**: Created a DVC pipeline that includes the following stages:

    o	Data Injection: Inject raw data into the system.
    o	Data Transformation: Clean and transform the data.
    o	Feature Engineering: Extract relevant features for model training.
    o	Model Building: Train and evaluate machine learning models.
    o	Model Evaluation: Evaluate model performance metrics.
    o	Model Registration: Register the best model for production use.
    o	Stage Transaction: Transition the model from staging to production.

### 3.Model Development

•	Developed a baseline **machine learning model** to classify emotions in text.
•	Model Evaluation: Regular performance tracking using **Dagshub** as the **MLflow tracking URI.**

•	Implemented model tests to ensure correct **model loading and inference behavior**

### 4.Flask API for Real-Time Inference

•	Built a **Flask** web application to serve the emotion detection model via a RESTful API for real-time inference.

### 5.Model Registration & Staging
•	Created separate scripts for **model registration** and **model staging** that move the best-performing model from staging to production.

### 6. CI/CD Pipeline with GitHub Actions

•	**Linting**: Linted all scripts using **flake8** to maintain code quality.
•	**Model Testing**: Automated model tests with **unittest**.
•	**DVC Pipeline Execution**: Ran the DVC pipeline using **dvc repro**.
•	**Model Promotion**: Automatically promoted the model to production after successful tests.

•	**Docker Build and Push**: Automated the Docker image build process and pushed the image to both **Docker Hub and AWS ECR.**

•	**Deployment**: Automated the deployment of the **Docker container** to **AWS EC2 instances.**

### 7. Containerization and Deployment

•	**Dockerization**: Containerized the Flask app along with the trained model into a Docker image.

•	**Deployment to EC2**: Deployed the Docker container to an **AWS EC2 instance** for scalable production hosting.

•	Docker images are pushed to both **Docker Hub** and **AWS ECR** for reliable version control and deployment.

•	Automated deployment from both **Docker Hub** and **AWS ECR** using **GitHub Actions.**

### GitHub Secrets

•	Managed sensitive variables and credentials securely using **GitHub Secrets**, including **AWS credentials**, **Docker Hub credentials**, and **Dagshub access tokens**.

## GitHub Actions Workflow

The **CI/CD pipeline** is defined using **GitHub Actions**, which automates all critical tasks such as testing, Docker image building, and deployment.
The pipeline is divided into the following jobs:

### 1.	Project Testing:
o	Lints the code.

o	Runs unit tests.

o	Executes the DVC pipeline to ensure data and model integrity.
### 2.	Docker Build & Push to AWS ECR:
o	Logs into AWS ECR.

o	Builds the Docker image.

o	Tags and pushes the image to AWS ECR for deployment.
### 3.	AWS EC2 Deployment:
o	Logs into AWS EC2 via SSH.

o	Pulls the latest Docker image from AWS ECR.

o	Deploys the container on AWS EC2.


## Secrets Configuration

To securely manage environment variables, the following secrets are required in your **GitHub repository:**


`AWS_ACCESS_KEY_ID`

`AWS_SECRET_ACCESS_KEY`

`AWS_REGION`

`AWS_ECR_REGISTRY`

`DAGSHUB_PAT`


`DOCKERHUB_USERNAME`

`DOCKERHUB_TOKEN`

`DOCKER_IMAGE_NAME`

`EC2_DEPLOYMENT_HOST`

`EC2_DEPLOYMENT_SSH_KEY`

`EC2_HOST`

`EC2_SSH_KEY`

`EC2_USER`



## How to Use it in Locally

### Step 1: Install Dependencies

To get started with the project, install the required Python dependencies:

pip install -r requirements.txt

Here is the command written in markdown format:

```bash
pip install requirements.txt
```

### Step 2: Initialize DVC

Initialize ***DVC*** to track your data:

```bash
DVC init
```

### Step 3: Run the Model Pipeline

To run the DVC pipeline and start training the model:

```bash
DVC repro
```
This will run all pipeline stages including data preprocessing, feature engineering, model training, and evaluation.

### Step 4: Run the Flask API Locally 
To run the Flask app locally:
1.	Build the Docker image:

```bash
docker build -t emotion-detection .
```



2.	Run the Docker container:

```bash
docker run -p 5000:5000 emotion-detection
```
if you are using DAGSHUB_PAT as variable then you should pass variable while running docker command like

```bash
docker run -e DAGSHUB_PAT=dagshub_token -p 5000:5000 emotion-detection
```


3.	Access the API by sending a POST request to
 http://localhost:5000/predict with your input text.
Example input:

```bash
"text": "I am so happy today!"
```


## 🔗 Links

[![](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hari-prasad-kale-896701236/)

