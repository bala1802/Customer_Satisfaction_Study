import logging
import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataStrategy, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple  

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated(pd.DataFrame, "X_train"),
    Annotated(pd.DataFrame, "X_test"),
    Annotated(pd.DataFrame, "y_train"),
    Annotated(pd.DataFrame, "y_test"),
]:
    """
    Data cleaning and pre-processing techniques are applied here

    Args:
        df: Raw Data
    
    Returns:
        X_train: Training Data
        X_test: Testing Data
        y_train: Testing label
        y_test: Testing label

    """

    try:
        pre_process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, pre_process_strategy)
        pre_processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(pre_processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning completed")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logging.error("Error in Cleaning the data: {}".format(e))
        raise e