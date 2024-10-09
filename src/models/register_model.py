import sys
import os
from src.custom_logging import logging
from src.exeption import CustomException
from src.utils import mlflow


dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
dagshub_url = "https://dagshub.com"
repo_owner = "kalehariprasad"
repo_name = "emotion-detection"
mlflow = mlflow()


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = mlflow.load_model_info(model_info_path)
        model_name = "my_model"
        mlflow.register_model(model_name, model_info)
    except Exception as e:
        logging.info(
            'Failed to complete the model registration process: %s', e
            )
        raise CustomException(e, sys)


if __name__ == '__main__':
    main()
