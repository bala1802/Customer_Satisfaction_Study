import logging
import pandas as pd
from zenml import step

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning and pre-processing techniques are applied here
    """
    pass