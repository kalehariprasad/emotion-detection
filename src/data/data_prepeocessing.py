# Data preprocessing

import os
import sys
import pandas as pd
from src.custom_logging import logging
from src.exeption import CustomException
from src.utils import TextNormalizer


text_processor = TextNormalizer()  # Corrected variable name for consistency


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('Data loaded properly')

        # Transform the data
        train_processed_data = text_processor.normalize_text(train_data)
        test_processed_data = text_processor.normalize_text(test_data)

        # Store the data inside data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(
            os.path.join(data_path, "train_processed.csv"), index=False
        )
        test_processed_data.to_csv(
            os.path.join(data_path, "test_processed.csv"), index=False
        )
        logging.info('Processed data  saved to %s', data_path)
    except Exception as e:
        logging.info('Failed to complete the data transformation : %s', e)
        raise CustomException(e, sys)


if __name__ == '__main__':
    main()
