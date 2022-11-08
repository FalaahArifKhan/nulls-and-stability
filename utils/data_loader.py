import pandas as pd
import numpy as np
from sys import getsizeof
from folktables import ACSDataSource, ACSEmployment

class ACSEmploymentDataset():
    def __init__(self, state, year, with_nulls=False, optimize=True):
        '''
        Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
        We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
        Alternatively, we could have altered the configuration from here:
        https://github.com/zykls/folktables/blob/main/folktables/acs.py
        '''
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features = ACSEmployment.features
        self.target = ACSEmployment.target
        self.categorical_columns = ['MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS', 'DEAR', 'DEYE', 'SEX', 'RAC1P', 'RELP', 'CIT', 'ANC','SCHL']
        self.numerical_columns = ['AGEP']

        if with_nulls==True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        if optimize==True:
            X_data = optimize_data_loading(X_data, self.categorical_columns)

        self.X_data = X_data
        self.y_data = acs_data[self.target].apply(lambda x: int(x == 1))

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        '''
        To save simulated nulls
        '''
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSDataset_from_demodq():
    ''' Following https://github.com/schelterlabs/demographic-data-quality '''
    def __init__(self, state, year, with_nulls=False, optimize=True):
        '''
        Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
        We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
        Alternatively, we could have altered the configuration from here:
        https://github.com/zykls/folktables/blob/main/folktables/acs.py
        '''
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features =  ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        self.target = ['PINCP']
        self.categorical_columns = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
        self.numerical_columns = ['AGEP', 'WKHP']

        if with_nulls==True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        if optimize==True:
            X_data = optimize_data_loading(X_data, self.categorical_columns)

        self.X_data = X_data
        self.y_data = acs_data[self.target].apply(lambda x: x >= 50000).astype(int)

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        '''
        To save simulated nulls
        '''
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


def optimize_data_loading(data, categorical):
    '''
    Optimizing the dataset size by downcasting categorical columns
    '''
    for column in categorical:
        data[column] = pd.to_numeric(data[column], downcast='integer')
    return data


class AdultDataset():

    def __init__(self):
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                   'hours-per-week', 'native-country', 'income-level']
        self.features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship','race', 'sex', 'native-country',
                        'age', 'hours-per-week', 'capital-gain', 'capital-loss']
        self.target = 'income-level'
        self.categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship','race', 'sex', 'native-country']
        self.numerical_columns = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']

        data_train = pd.read_csv('data/adult-income/adult.data', na_values=['?'], header=None, names=columns, sep=", ")
        data_test = pd.read_csv('data/adult-income/adult.test', na_values=['?'], header=None, names=columns, sep=", ")
        data = pd.concat([data_train, data_test])
        # Remove nonsensical record
        data = data[data.age != '|1x3 Cross validator']
        data.age = pd.to_numeric(data.age)

        self.X_data = data[self.features]
        self.y_data = data[self.target].apply(lambda x: int((x == '>50K') | (x == '>50K.')))

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

