"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.1
"""
import numpy as np
import pandas as pd

import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

def split_data(data, parameters):
    X = create_train(data)
    y = create_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    regressor = xgb.XGBRegressor(learning_rate=0.05, max_depth=3,  n_estimators=2000, nthread=-1)
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)

def create_target(train):
    y_train = np.log1p(train["SalePrice"])
    return y_train


def create_train(train):
    return train.drop('SalePrice', axis=1)