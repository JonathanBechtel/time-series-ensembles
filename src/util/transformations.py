"""_summary_: This file contains functions that transform data in a variety of ways.
    Usually used to inverse transformations on predictions to calculate error metrics.
"""
import numpy as np
import pandas as pd


def transform_interval_difference(
    df, target_col="Target", value_col="Value", interval=7
):
    """_summary_

    Args:
        df (DataFrame): input dataframe with target and value columns
        target_col (str, optional): _description_. Defaults to 'Target'.
        value_col (str, optional): _description_. Defaults to 'Value'.
        interval (int, optional): _description_. Defaults to 7.

    Returns:
        transformed: transformed values
    """

    transformed = df[target_col].copy()

    transformed = transformed.fillna(df[value_col])

    for i in range(interval, len(transformed)):
        transformed.iloc[i] += transformed.iloc[i - interval]

    return transformed


def transform_target_prediction(
    df, transform_function: callable, entity_col="series", **kwargs
):
    unique_entities = df[entity_col].unique()

    final_vals = []

    for entity in unique_entities:
        temp_df = df.loc[df[entity_col] == entity].copy()

        final_vals.append(transform_function(temp_df, **kwargs))

    return pd.concat(final_vals)


def transform_log_difference(df, target_col="target", value_col="value"):
    transformed = df[target_col].copy()
    transformed.iloc[0] = np.log1p(df[value_col].iloc[0])
    # transformed = transformed.fillna(np.log1p(df[value_col]))
    transformed = transformed.cumsum()
    transformed = np.expm1(transformed)

    return transformed


def transform_difference(df, target_col="target", value_col="value"):
    transformed = df[target_col].copy()
    transformed.iloc[0] = df[value_col].iloc[0]
    # transformed = transformed.fillna(df[value_col])
    transformed = transformed.cumsum()

    return transformed


def transform_double_difference(df, target_col="target", value_col="value"):
    first_value = df[value_col].iloc[0]
    first_diff = df[value_col].iloc[1] - first_value

    transformed = df[target_col].copy()

    second_diff_reconstructed = transformed.cumsum() + first_diff
    first_diff_reconstructed = (
        second_diff_reconstructed.cumsum() + first_value + first_diff
    )

    return first_diff_reconstructed


def transform_log_double_difference(df, target_col="target", value_col="value"):
    first_value = np.log1p(df[value_col].iloc[0])
    first_diff = np.log1p(df[value_col].iloc[1]) - first_value

    transformed = df[target_col].copy()

    second_diff_reconstructed = transformed.cumsum() + first_diff
    first_diff_reconstructed = (
        second_diff_reconstructed.cumsum() + first_value + first_diff
    )

    return np.expm1(first_diff_reconstructed)
