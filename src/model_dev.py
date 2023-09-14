import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all the models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train: Training data
            y_train: Testing data
        
        Returns:
            None
        """
        return None
    
class LinearRegressionModel(Model):
    
    def train(self, X_train, y_train, **kwargs):
        
        """
        Trains the Linear Regression Model

        Args:
            X_train: Training data
            y_train: Testing data
        
        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e