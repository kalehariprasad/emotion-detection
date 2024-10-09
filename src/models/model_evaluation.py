import sys
import os
from src.custom_logging import logging
from src.exeption import CustomException
from src.utils import Model
from src.utils import DataHandler
import mlflow


dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
repo_owner = "kalehariprasad"
repo_name = "emotion-detection"
dagshub_url = f"https://dagshub.com/{repo_owner}/{repo_name}"
mlflow.set_tracking_uri(dagshub_url)
model = Model()


def main():
    mlflow.set_experiment("dvc-pipeline-github-actions")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            clf = model.load_model('./models/model/model.pkl')
            test_data = data_handler.load_data('./data/processed/test_bow.csv')
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
            metrics = model.evaluate_model(clf, X_test, y_test)
            model.save_metrics(metrics, 'reports/metrics.json')
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "model")
            # Save model info
            model.save_model_info(
                run.info.run_id, "model", 'reports/experiment_info.json'
                )
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')
            # Log the model info file to MLflow
            mlflow.log_artifact('reports/experiment_info.json')
            # Log the evaluation errors log file to MLflow
            # mlflow.log_artifact('model_evaluation_errors.log')
        except Exception as e:
            logging.info(
                'Failed to complete the model evaluation process: %s', e
                )
            raise CustomException(e, sys)


if __name__ == '__main__':
    main()
