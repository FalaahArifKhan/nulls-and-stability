import pandas as pd
import numpy as np

def nulls_simulator(data, target_col, condition_col, special_values, fraction, nan_value=np.nan):
    """
    Description: simulate nulls for the target column in the dataset based on the condition column and its special values.
    Input:
    :param data: a pandas dataframe, in which nulls values should be simulated
    :param target_col: a column in the dataset, in which nulls should be placed
    :param condition_col: a column in the dataset based on which null location should be identified
    :param special_values: list of special values for the condition column; special_values and condition_col state the condition,
        where nulls should be placed
    :param fraction: float in range [0.0, 1.0], fraction of nulls, which should be placed based on the condition
    :param nan_value: a value, which should be used as null to be placed in the dataset
    Output: a dataset with null values based on the condition and fraction
    """
    if target_col not in data.columns:
        return ValueError(f'Column {target_col} does not exist in the dataset')
    if condition_col not in data.columns:
        return ValueError(f'Column {condition_col} does not exist in the dataset')
    if not (0 <= fraction <= 1):
        return ValueError(f'Fraction {fraction} is not in range [0, 1]')

    dataset = data.copy(deep=True)
    corrupted_data = data[data[condition_col].isin(special_values)]
    #null_idx = corrupted_data.index.to_list()
    rows_to_corrupt = get_sample_rows(corrupted_data, target_col, fraction).to_list()
    dataset.loc[rows_to_corrupt, [target_col]] = nan_value
    #corrupted_data.loc[rows_to_corrupt, [target_col]] = nan_value
    #dataset.iloc[null_idx] = corrupted_data
    return dataset

def get_sample_rows(data, target_col, fraction):
    """
    Description: create a list of random indexes for rows, which will be used to place nulls in the dataset
    """
    n_values_to_discard = int(len(data) * min(fraction, 1.0))
    perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
    perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)

    depends_on_col = np.random.choice(list(set(data.columns) - {target_col}))
    # Pick a random percentile of values in other column
    rows = data[depends_on_col].sort_values().iloc[perc_idx].index
    return rows