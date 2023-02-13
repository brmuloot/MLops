import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        # A COMPPLETER
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        
    def fit(self, X, y=None):
        # A COMPLETER 
        return self
    
    def transform(self, X, y=None):
        # Convert to radians
        X[self.start_lat] = np.radians(X[self.start_lat])
        X[self.start_lon] = np.radians(X[self.start_lon])
        X[self.end_lat] = np.radians(X[self.end_lat])
        X[self.end_lon] = np.radians(X[self.end_lon])

        dlat = X[self.end_lat] - X[self.start_lat]
        dlon = X[self.end_lon] - X[self.start_lon]

        # Calculate the haversine distance
        a = np.sin(dlat/2)**2 + np.cos(X[self.start_lat]) * np.cos(X[self.end_lat]) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        X['distance'] = 6371 * c
        
        return X[['distance']]

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self, time_column='pickup_datetime'):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.time_column] = pd.to_datetime(X_[self.time_column])
        X_['dow'] = X_[self.time_column].dt.dayofweek
        X_['hour'] = X_[self.time_column].dt.hour
        X_['month'] = X_[self.time_column].dt.month
        X_['year'] = X_[self.time_column].dt.year
        return X_[['dow', 'hour', 'month', 'year']]