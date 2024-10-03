import os
import sys
import pandas as pd
import numpy as np
import yaml
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from src.custom_logging import logging
from src.exeption import CustomException
nltk.download('wordnet')
nltk.download('stopwords')


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

    def save_data(
        self, data: pd.DataFrame, file_path: str
    ) -> None:
        """Save the train and test datasets."""
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data.to_csv(file_path, index=False)
            logging.info('Processed data  saved to %s', file_path)
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
