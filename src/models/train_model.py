import sys
from src.custom_logging import logging
from src.exeption import CustomException
from src.utils import Model
from src.utils import DataHandler


data_handler = DataHandler(params_path='params.yaml')
model = Model()


def main():
    try:
        train_data = data_handler.load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        clf = model.train_model(X_train, y_train)
        data_handler.save_object(clf, 'models/model/model.pkl')
    except Exception as e:
        logging.info('Failed to complete the model building process: %s', e)
        raise CustomException(e, sys)


if __name__ == '__main__':
    main()
