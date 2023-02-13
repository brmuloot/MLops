import numpy as np
import pandas as pd
import encoders
import utils
import data

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self, X, y, pipeline=None):
        self.X = X
        self.y = y
        self.pipeline = None

    def set_pipeline(self):
        distance_pipeline = Pipeline([
            ('distance_transformer', encoders.DistanceTransformer()),
            ('standard_scaler', StandardScaler())
            ])

        time_pipeline = Pipeline([
            ('time_encoder', encoders.TimeFeaturesEncoder()),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

        preprocessing_pipeline = ColumnTransformer(
            transformers=[
            ('distance', distance_pipeline, ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']),
            ('time_features', time_pipeline, ['pickup_datetime'])
            ])

        pipe = Pipeline(steps=[('preproc', preprocessing_pipeline),
                    ('linear_model', LinearRegression())])

        self.pipeline = pipe

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        rmse = utils.compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

df = data.get_data(nrows=1000)
X = df.drop(columns = "fare_amount")
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

model = Trainer(X_train, y_train)
model.set_pipeline()
model.train(X_train, y_train)
model.evaluate(X_test, y_test)
