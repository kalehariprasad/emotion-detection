
import os
import sys
import pandas as pd
import numpy as np
import yaml
import re
import string
import nltk
import json
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score
    )
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from src.custom_logging import logging
from src.exeption import CustomException
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


class DataHandler:
    def __init__(self, params_path: str):
        self.params_path = params_path
        self.params = self.load_params()

    def load_params(self) -> dict:
        """Load parameters from a YAML file."""
        try:
            with open(self.params_path, 'r') as file:
                params = yaml.safe_load(file)
            logging.info('Parameters retrieved from %s', self.params_path)
            return params
        except FileNotFoundError:
            logging.info('File not found: %s', self.params_path)
            raise
        except yaml.YAMLError as e:
            logging.info('YAML error: %s', e)
            raise
        except Exception as e:
            logging.info('Unexpected error: %s', e)
            raise CustomException(e, sys)

    def load_data(self, data_url: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(data_url)
            logging.info('Data loaded from %s', data_url)
            # Calculate the percentage of null values in each column
            null_percentage = df.isnull().mean() * 100
            logging.info('Null value percentages:\n%s', null_percentage)
            # Drop rows with null values if they are less than 5%
            if null_percentage.max() < 5:
                df = df.dropna()
                logging.info(
                    'Dropped rows with null values as they were less than 5% .'
                    )
            else:
                logging.warning(
                    'Null values exceed 5%, not dropping any rows.'
                    )
            return df
        except Exception as e:
            logging.info(
                'Unexpected error occurred while loading data: %s', e)
            raise CustomException(e, sys)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        try:
            pd.set_option('future.no_silent_downcasting', True)
            df = df.drop(columns=['tweet_id'], errors='ignore')
            final_df = df[df['sentiment'].isin(
                ['happiness', 'sadness']
            )].copy()
            final_df['sentiment'] = final_df['sentiment'].replace(
                {'happiness': 1, 'sadness': 0}
            )
            logging.info('Data preprocessing completed')
            return final_df
        except KeyError as e:
            logging.error('Missing column in the dataframe: %s', e)
            raise CustomException(e, sys)
        except Exception as e:
            logging.info('Unexpected error during preprocessing: %s', e)
            raise CustomException(e, sys)

    def save_object(
        self, object, file_path: str
    ) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as file:
                pickle.dump(object, file)
        except Exception as e:
            logging.info(
                "unexpected error occured while saving object"
            )
            raise CustomException(e, sys)

    def save_data(
        self, data: pd.DataFrame, file_path: str
    ) -> None:
        """Save the train and test datasets."""
        try:
<<<<<<< HEAD
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data.to_csv(file_path, index=False)
            logging.info('Processed data  saved to %s', file_path)
=======
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data.to_csv(file_path, index=False)
            logging.info('Data saved to %s', file_path)
>>>>>>> 2fe5de3d94f2ca2b16301d96a5f4b688bc958dd0
        except Exception as e:
            logging.info(
                'Unexpected error occurred while saving the data: %s', e)
            raise CustomException(e, sys)


class TextNormalizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def lemmatization(self, text):
        try:
            """Lemmatize the text."""
            text = text.split()
            text = [self.lemmatizer.lemmatize(word) for word in text]
            return " ".join(text)
        except Exception as e:
            logging.info(
                'Unexpected error occurred during lemmatization: %s', e
            )
            raise CustomException(e, sys)

    def remove_stop_words(self, text):
        try:
            """Remove stop words from the text."""
            text = [
                word for word in str(text).split()
                if word not in self.stop_words
            ]
            return " ".join(text)
        except Exception as e:
            logging.info(
                'Unexpected error occurred while removing stop words: %s', e
            )
            raise CustomException(e, sys)

    def removing_numbers(self, text):
        try:
            """Remove numbers from the text."""
            text = ''.join([char for char in text if not char.isdigit()])
            return text
        except Exception as e:
            logging.info(
                'Unexpected error occurred while removing numbers: %s', e
            )
            raise CustomException(e, sys)

    def lower_case(self, text):
        try:
            """Convert text to lower case."""
            text = text.split()
            text = [word.lower() for word in text]
            return " ".join(text)
        except Exception as e:
            logging.info(
                'error occurred while converting to lower case: %s', e
            )
            raise CustomException(e, sys)

    def removing_punctuations(self, text):
        try:
            """Remove punctuations from the text."""
            text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
            text = text.replace('؛', "")
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            logging.info(
                'Unexpected error occurred while removing punctuations: %s', e
            )
            raise CustomException(e, sys)

    def removing_urls(self, text):
        try:
            """Remove URLs from the text."""
            url_pattern = re.compile(r'https?://\S+|www\.\S+')
            return url_pattern.sub(r'', text)
        except Exception as e:
            logging.info(
                'Unexpected error occurred while removing URLs: %s', e
            )
            raise CustomException(e, sys)

    def remove_small_sentences(self, df):
        try:
            """Remove sentences with less than 3 words."""
            for i in range(len(df)):
                if len(df.text.iloc[i].split()) < 3:
                    df.text.iloc[i] = np.nan
        except Exception as e:
            logging.info(
                'error occurred while removing small sentences: %s', e
            )
            raise CustomException(e, sys)

    def normalize_text(self, df):
        """Normalize the text data."""
        try:
            df['content'] = df['content'].apply(self.lower_case)
            logging.info('Converted to lower case')
            df['content'] = df['content'].apply(self.remove_stop_words)
            logging.info('Stop words removed')
            df['content'] = df['content'].apply(self.removing_numbers)
            logging.info('Numbers removed')
            df['content'] = df['content'].apply(self.removing_punctuations)
            logging.info('Punctuations removed')
            df['content'] = df['content'].apply(self.removing_urls)
            logging.info('URLs removed')
            df['content'] = df['content'].apply(self.lemmatization)
            logging.info('Lemmatization performed')
            logging.info('Text normalization completed')
            return df
        except Exception as e:
            logging.info(
                'Unexpected Error during text normalization: %s', e
            )
            raise CustomException(e, sys)


class Preprocessing:
    def __init__(self):
        pass

    def apply_bow(
            self, train_data: pd.DataFrame, test_data: pd.DataFrame,
            max_features: int
            ) -> tuple:
        """Apply Count Vectorizer to the data."""
        try:
            vectorizer = CountVectorizer(max_features=max_features)
            X_train = train_data['content'].values
            y_train = train_data['sentiment'].values
            X_test = test_data['content'].values
            y_test = test_data['sentiment'].values
            X_train_bow = vectorizer.fit_transform(X_train)
            X_test_bow = vectorizer.transform(X_test)
            train_df = pd.DataFrame(X_train_bow.toarray())
            train_df['label'] = y_train
            test_df = pd.DataFrame(X_test_bow.toarray())
            test_df['label'] = y_test
            os.makedirs(os.path.dirname(
                'models/objects/vectorizer.pkl'), exist_ok=True
                )
            pickle.dump(
                vectorizer, open('models/objects/vectorizer.pkl', 'wb')
                )
            logging.info('Bag of Words applied and data transformed')
            return train_df, test_df
        except Exception as e:
            logging.info('Error during Bag of Words transformation: %s', e)
            raise CustomException(e, sys)


class Model:
    def __init__(self) -> None:
        pass

    def train_model(
            self, X_train: np.ndarray, y_train: np.ndarray
            ) -> LogisticRegression:
        try:
            clf = LogisticRegression(
                C=1, solver='liblinear', penalty='l2'
                )
            clf.fit(X_train, y_train)
            logging.info('Model training completed')
            return clf
        except Exception as e:
            logging.info(
                'Unexpected error occurred while trainig: %s', e
                )
            raise CustomException(e, sys)

    def load_model(self, file_path: str):
        """Load the trained model from a file."""
        try:
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            logging.info('Model loaded from %s', file_path)
            return model
        except FileNotFoundError:
            logging.info('File not found: %s', file_path)
            raise
        except Exception as e:
            logging.info(
                'Unexpected error occurred while loading the model: %s', e
                )
            raise CustomException(e, sys)

    def evaluate_model(
            self, clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model and return the evaluation metrics."""
        try:
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)

            metrics_dict = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
            logging.info('Model evaluation metrics calculated')
            return metrics_dict
        except Exception as e:
            logging.info('Error during model evaluation: %s', e)
            raise CustomException(e, sys)

    def save_metrics(self, metrics: dict, file_path: str) -> None:
        """Save the evaluation metrics to a JSON file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                json.dump(metrics, file, indent=4)
            logging.info('Metrics saved to %s', file_path)
        except Exception as e:
            logging.info('Error occurred while saving the metrics: %s', e)
            raise CustomException(e, sys)

    def save_model_info(
            self, run_id: str, model_path: str, file_path: str
            ) -> None:
        """Save the model run ID and path to a JSON file."""
        try:
            model_info = {'run_id': run_id, 'model_path': model_path}
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                json.dump(model_info, file, indent=4)
            logging.info('Model info saved to %s', file_path)
        except Exception as e:
            logging.info('Error occurred while saving the model info: %s', e)
            raise CustomException(e, sys)


class MLFlowInstance:
    def __init__(self):
        pass

    def load_model_info(self, file_path: str) -> dict:
        """Load the model info from a JSON file."""
        try:
            with open(file_path, 'r') as file:
                model_info = json.load(file)
            logging.info('Model info loaded from %s', file_path)
            return model_info
        except FileNotFoundError:
            logging.info('File not found: %s', file_path)
            raise
        except Exception as e:
            logging.info(
                'Unexpected error occurred while loading the model info: %s', e
                )
            raise CustomException(e, sys)

    def register_model(self,  model_name: str, model_info: dict):
        """Register the model to the MLflow Model Registry."""
        try:
            model_uri = (
                f"runs:/{model_info['run_id']}/"
                f"{model_info['model_path']}"
            )
            # Register the model
            model_version = mlflow.register_model(model_uri, model_name)
            # Transition the model to "Staging" stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            logging.info(
                f'Model {model_name} version {model_version.version} .'
            )
        except Exception as e:
            logging.info('Error during model registration: %s', e)
            raise CustomException(e, sys)

    def get_latest_model_version(self, model_name: str) -> str:
        """Get the latest version of the model."""
        client = mlflow.MlflowClient()
        registered_models = client.search_registered_models(
            filter_string=f"name='{model_name}'"
            )
        if registered_models:
            model = registered_models[0]
            if model.latest_versions:
                latest_version = max(
                    int(version.version) for version in model.latest_versions
                    )
                logging.info(
                    f'Latest version of {model_name} is {latest_version}.'
                    )
                return str(latest_version)
        logging.info(f'No versions found for model {model_name}.')
        return None

    def transfer_stage_to_production(self, model_name: str):
        """Transfer the model from Staging to Production."""
        try:
            client = mlflow.tracking.MlflowClient()
            latest_version = self.get_latest_model_version(model_name)
            if latest_version:
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage="Production"
                )
                logging.info(
                    f'Model {model_name} {latest_version} moved to Production.'
                    )
            else:
                logging.info(
                    f'No model version to transition for {model_name}.'
                    )
        except Exception as e:
            logging.info('Error during model stage transfer: %s', e)
            raise CustomException(e, sys)
