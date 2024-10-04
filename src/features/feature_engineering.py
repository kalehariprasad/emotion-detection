import os
import sys
from src.custom_logging import logging
from src.exeption import CustomException
from src.utils import Preprocessing
from src.utils import DataHandler


data_handler = DataHandler(params_path='params.yaml')
preprocessor = Preprocessing()


def main():
    try:
        params = data_handler.load_params()
        max_features = params['feature_engineering']['max_features']

        train_data = data_handler.load_data(
            './data/interim/train_processed.csv'
            )
        test_data = data_handler.load_data(
            './data/interim/test_processed.csv'
            )
        train_df, test_df = preprocessor.apply_bow(
            train_data, test_data, max_features
            )
        data_handler.save_data(
            train_df, os.path.join("./data", "processed", "train_bow.csv")
            )
        data_handler.save_data(
            test_df, os.path.join("./data", "processed", "test_bow.csv")
            )
        data_handler.save_object(
                preprocessor, os.path.join("./models", "preprocessor.pkl")
        )
    except Exception as e:
        logging.info(
            'Failed to complete the feature engineering process: %s', e
            )
        raise CustomException(e, sys)


if __name__ == '__main__':
    main()
