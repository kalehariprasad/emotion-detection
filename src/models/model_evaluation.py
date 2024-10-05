import sys
from src.custom_logging import logging
from src.exeption import CustomException
from src.utils import Model
from src.utils import DataHandler
import mlflow
import dagshub

mlflow.set_tracking_uri(
    'https://dagshub.com/kalehariprasad/emotion-detection.mlflow'
    )
dagshub.init(
    repo_owner='kalehariprasad', repo_name='emotion-detection', mlflow=True
    )
data_handler = DataHandler(params_path='params.yaml')
model = Model()


def main():
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            clf = model.load_model('./models/model.pkl')
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
            mlflow.log_artifact('reports/model_info.json')
            # Log the evaluation errors log file to MLflow
            mlflow.log_artifact('model_evaluation_errors.log')
        except Exception as e:
            logging.info(
                'Failed to complete the model evaluation process: %s', e
                )
            raise CustomException(e, sys)


if __name__ == '__main__':
    main()