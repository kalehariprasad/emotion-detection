from src.exeption import CustomException
from src.custom_logging import logging
import os
import sys

try:
    a = 3
    b = "5"
    c = a + b  # This will raise a TypeError
    print(c)
except Exception as e:
    logging.info(f'error occured while performing addition with message {e}')
    raise CustomException(e, sys)
