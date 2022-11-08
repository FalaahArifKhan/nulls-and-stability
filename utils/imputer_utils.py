import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

def find_column_mode(data):
    result = stats.mode(data)
    return result.mode[0]


def find_column_mean(data):
    return np.mean(data).round()


def find_column_median(data):
    return np.median(data).round()

def base_regressor(column_type):
        if column_type == 'numerical':
            model = LinearRegression()
        elif column_type == 'categorical':
            model = LogisticRegression()
        else:
            raise ValueError(
                    "Can only support numerical or categorical columns, got column_type={0}".format(column_type))
            model = None
        return model

def base_knn(column_type, n_neighbors):
        if column_type == 'numerical':
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        elif column_type == 'categorical':
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            raise ValueError(
                    "Can only support numerical or categorical columns, got column_type={0}".format(column_type))
            model = None
        return model