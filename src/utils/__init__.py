import os
import sys
import pandas as pd
from src.custom_logging import logging
from src.exeption import CustomException
import yaml


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
            return df
        except pd.errors.ParserError as e:
            logging.info('Failed to parse the CSV file: %s', e)
            raise CustomException(e, sys)
        except Exception as e:
            logging.info(
                'Unexpected error occurred while loading data: %s', e)
            raise CustomException(e, sys)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        try:
            pd.set_option('future.no_silent_downcasting', True)
            df = df.drop(columns=['tweet_id'], errors='ignore')
            final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
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

    def save_data(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str
    ) -> None:
        """Save the train and test datasets."""
        try:
            raw_data_path = os.path.join(data_path, 'raw')
            os.makedirs(raw_data_path, exist_ok=True)
            train_data.to_csv(
                os.path.join(raw_data_path, "train.csv"), index=False
            )
            test_data.to_csv(
                os.path.join(raw_data_path, "test.csv"), index=False
            )
            logging.info(
                'Train and test data saved to %s', raw_data_path
            )
        except Exception as e:
            logging.info(
                'Unexpected error occurred while saving the data: %s', e)
            raise CustomException(e, sys)
