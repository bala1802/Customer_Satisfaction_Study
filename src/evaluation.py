import logging

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for the model evaluation
    """

    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculates the scores for the trained model

        Args:
            y_true: True Labels
            y_pred: Predicted Labels
        
        Returns:
            None
        """
        
        pass

class MSE(Evaluation):
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true=y_true, y_pred=y_pred)   
            logging.info("MSE: {}".format(mse))
        except Exception as e:
            logging.error('Error in calculating MSE: {}'.format(e))
            raise e

class R2Score(Evaluation):

    """
    Evaluation strategy that uses R2 Score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating R2 Square')
            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.info('Error in calculating the R2 score: {}'.format(e))
            raise e

class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Square 
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE Value")
            rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
            logging.info("RMSE: {}".format(rmse))
        except Exception as e:
            logging.error("Error is calculating the RMSE Score: {}".format(e))
            raise e