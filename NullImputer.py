import pandas as pd
import numpy as np
from scipy import stats

from utils.imputer_utils import *

class NullImputer():
    def __init__(self, target_columns, how="mean", trimmed=0, conditional_column=None, special_value=None):
        self.how = how
        self.target_columns = target_columns
        self.conditional_column = conditional_column
        self.trimmed = trimmed
        self.mask = None
        values_to_impute = {}
        for col in self.target_columns:
            values_to_impute[col] = special_value
        self.values_to_impute = values_to_impute

    def fit(self, X, y=None):
        allowed = ["mean", "median", "mode", "special"]
        
        if self.how not in allowed:
            raise ValueError(
                "Can only use these strategies: {0}  got strategy={1}".format(
                    allowed, self.how
                )
            )
            return self.data

        data = X.copy()

        if self.how == 'special':
            if self.values_to_impute[self.target_columns[0]] !=None :
                return
            else:
                raise ValueError("Special value was not passed during initialization") 

        elif self.how == 'mode':
            get_impute_value = find_column_mode
        elif self.how == 'mean':
            get_impute_value = find_column_mean
        elif self.how == 'median':
            get_impute_value = find_column_median

        values_to_impute = dict()
        for col in self.target_columns:
            filtered_df = data[~data[col].isnull()].copy(deep=True)

            # Trimming rows
            if self.trimmed>0:
                reduce_n_rows = int(filtered_df.shape[0] / 100 * self.trimmed)
                filtered_df.sort_values(by=[col], ascending=False, inplace=True)
                filtered_df = filtered_df[reduce_n_rows: -reduce_n_rows]

            if self.conditional_column is None:
                values_to_impute[col] = get_impute_value(filtered_df[col].values)

            else:
                # Imputing conditioned on another column -- using a mask for this
                mapping_dict = dict()
                for val in filtered_df[self.conditional_column].unique():
                    fillna_val = get_impute_value(filtered_df[filtered_df[self.conditional_column] == val][col].values)
                    #print(f"Need to impute {col} with value {fillna_val}, where {self.conditional_column} == {val}")
                    mapping_dict[val] = fillna_val
                values_to_impute[col] = mapping_dict

        self.values_to_impute = values_to_impute
        return

    def transform(self, X, y=None):
        data = X.copy()
        if self.values_to_impute is None:
            raise ValueError(
                "Call fit before calling transform!"
            )
            return data

        if self.conditional_column is None:
            for col in self.target_columns:
                data[col].fillna(value=self.values_to_impute[col], inplace=True)

        else:
            for col in self.target_columns:
                missing_mask = data[col].isna()
                data.loc[missing_mask, col] = data.loc[missing_mask, self.conditional_column].map(self.values_to_impute[col])

        return data


    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

            








