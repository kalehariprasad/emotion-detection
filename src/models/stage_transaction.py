import sys
import os
import mlflow
from src.custom_logging import logging
from src.exeption import CustomException
from src.utils import MLFlowInstance


dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
dagshub_url = "https://dagshub.com"
repo_owner = "kalehariprasad"
repo_name = "emotion-detection"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
mlflow_instance = MLFlowInstance()


def main():
    try:
        model_name = "my_model"
        mlflow_instance.transfer_stage_to_production(model_name)
    except Exception as e:
        logging.info(
            'Failed to complete the model transaction process: %s', e
            )
        raise CustomException(e, sys)


if __name__ == '__main__':
    main()
