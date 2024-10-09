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
repo_owner = "kalehariprasad"
repo_name = "emotion-detection"
dagshub_url = f"https://dagshub.com/{repo_owner}/{repo_name}"
mlflow.set_tracking_uri(dagshub_url)
mlflow_instance = MLFlowInstance()


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = mlflow_instance.load_model_info(model_info_path)
        model_name = "my_model"
        logging.info(f'model info is { model_info}')
        mlflow_instance.register_model(model_name, model_info)
    except Exception as e:
        logging.info(
            'Failed to complete the model registration process: %s', e
            )
        raise CustomException(e, sys)


if __name__ == '__main__':
    main()
