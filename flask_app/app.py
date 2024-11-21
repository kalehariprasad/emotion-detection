# updated app.py
from flask import Flask, render_template, request
import mlflow
import pickle
import pandas as pd
from src.custom_logging import logging
from src.exeption import CustomException
from src.utils import TextNormalizer
import os
import sys

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_PAT environment variable is not set"
        )
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
dagshub_url = "https://dagshub.com"
repo_owner = "kalehariprasad"
repo_name = "emotion-detection"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(
    f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow'
    )

app = Flask(__name__)
text_normalizer = TextNormalizer()


# load model from model registry
def get_latest_model_version(model_name):
    try:
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(
            model_name, stages=["Production"]
            )
        if not latest_version:
            latest_version = client.get_latest_versions(
                model_name, stages=["None"]
                )
        return latest_version[0].version if latest_version else None
    except Exception as e:
        logging.info(
            'Failed to load latest model: %s', e
            )
        raise CustomException(e, sys)


model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

preprocessor = pickle.load(open('models/objects/vectorizer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']
    df = pd.DataFrame({'content': [text]})
    normalized_df = text_normalizer.normalize_text(df)
    normalized_text = normalized_df['content'].iloc[0]
    # bow
    features = preprocessor.transform([normalized_text])
    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(
        features.toarray(), columns=[str(i) for i in range(features.shape[1])]
        )
    # prediction
    result = model.predict(features_df)
    # show
    return render_template('index.html', result=result[0])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
