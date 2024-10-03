import sys
import os
from sklearn.model_selection import train_test_split
from src.custom_logging import logging
from src.exeption import CustomException
from src.utils import DataHandler


url = (
    'https://raw.githubusercontent.com/campusx-official/'
    'jupyter-masterclass/main/tweet_emotions.csv'
)


def main():
    try:
        data_handler = DataHandler(params_path='params.yaml')
        test_size = data_handler.params['data_ingestion']['test_size']
        df = data_handler.load_data(data_url=url)
        final_df = data_handler.preprocess_data(df)
        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )
        data_path = os.path.join(os.getcwd(), "data")
        raw_data_path = os.path.join(data_path, "raw")
        train_file_path = os.path.join(raw_data_path, "train.csv")
        test_file_path = os.path.join(raw_data_path, "test.csv")
        data_handler.save_data(train_data, train_file_path)
        data_handler.save_data(test_data, test_file_path)
    except Exception as e:
        logging.info('Failed to complete the data ingestion process: %s', e)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
