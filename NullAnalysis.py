import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from utils.analysis_utils import *
from NullImputer import *

class NullAnalysis():
    def __init__(self, dataset, protected_groups, priv_values):
        self.features = dataset.features
        self.target = dataset.target
        self.categorical_columns = dataset.categorical_columns
        self.numerical_columns = dataset.numerical_columns
        self.X_data = dataset.X_data
        self.y_data = dataset.y_data
        self.protected_groups = protected_groups
        self.priv_values = priv_values
        self.columns_with_nulls = dataset.columns_with_nulls
        self.columns_without_nulls = list(set(self.features) - set(self.columns_with_nulls)) #For NullPredictors
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.groups = None
        self.base_model = None
        self.bootstrap_results = {}

    def create_train_test_val_split(self, SEED):
        X_, X_test, y_, y_test = train_test_split(self.X_data, self.y_data, test_size=0.2, random_state=SEED)
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.25, random_state=SEED)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.groups = set_protected_groups(self.X_test, self.protected_groups, self.priv_values)
        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val

    def set_train_test_val_data_by_index(self, train_idx, test_idx, val_idx):
        self.X_train = self.X_data.loc[train_idx]
        self.y_train = self.y_data.loc[train_idx]
        self.X_test = self.X_data.loc[test_idx]
        self.y_test = self.y_data.loc[test_idx]
        self.X_val = self.X_data.loc[val_idx]
        self.y_val = self.y_data.loc[val_idx]
        self.groups = set_protected_groups(self.X_test, self.protected_groups, self.priv_values)
        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val

    def fit_ensemble_with_bootstrap(self, base_model, n_estimators, boostrap_size, with_replacement=True):
        '''
        Quantifying uncertainty of predictive model by constructing an ensemble from boostrapped samples
        '''
        predictions = {}
        ensemble = {}
        
        for m in range(n_estimators):
            '''
            encoder = ColumnTransformer(transformers=[
                ('categorical_features', OneHotEncoder(categories=[list(set(self.X_train[col])) for col in self.categorical_columns], sparse=False), self.categorical_columns),
                ('numerical_features', StandardScaler(), self.numerical_columns)])
            pipeline = Pipeline([
                            ('features', encoder),
                            ('learner', base_model)
                        ])
            model = pipeline
            '''
            model = base_model
            X_sample, y_sample = generate_bootstrap(self.X_train, self.y_train, boostrap_size, with_replacement)
            model.fit(pd.DataFrame(X_sample, columns=self.X_train.columns), y_sample.ravel())
            predictions[m] = model.predict_proba(self.X_test)[:, 0]

            ensemble[m] = model
        return ensemble, predictions


    def fit_ensemble_with_bootstrap_drop_rows(self, base_model, n_estimators, boostrap_size, with_replacement=True):
        '''
        Quantifying uncertainty of predictive model by constructing an ensemble from boostrapped samples
        '''
        predictions = {}
        ensemble = {}
        
        for m in range(n_estimators):
            '''
            encoder = ColumnTransformer(transformers=[
                ('categorical_features', OneHotEncoder(categories=[list(set(self.X_train[col])) for col in self.categorical_columns], sparse=False), self.categorical_columns),
                ('numerical_features', StandardScaler(), self.numerical_columns)])
            pipeline = Pipeline([
                            ('features', encoder),
                            ('learner', base_model)
                        ])
            model = pipeline
            '''
            model = base_model
            X_sample, y_sample = generate_bootstrap(self.X_train, self.y_train, boostrap_size, with_replacement)
            
            X_train_without_nulls = pd.DataFrame(X_sample, columns=self.X_train.columns).copy(deep=True)
            X_train_without_nulls.dropna(inplace=True)
            y_train_without_nulls = y_sample[X_train_without_nulls.index]

            test_imputer = NullImputer(self.columns_with_nulls, how="mode", trimmed=0, conditional_column=None)
            test_imputer.fit(X_train_without_nulls)
            X_test_without_nulls = test_imputer.transform(self.X_test)

            model.fit(X_train_without_nulls, y_train_without_nulls)
            predictions[m] = model.predict_proba(X_test_without_nulls)[:, 0]

            ensemble[m] = model
        return ensemble, predictions
