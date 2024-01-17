"""OpenML DataFrame Module

This module defines the `OpenMLDataFrame` class, a custom class for datasets retrieved from OpenML as dataframes.

"""

import logging

import openml

from .tabular_dataframe import TabularDataFrame

logger = logging.getLogger(__name__)

exceptions_binary = [45062]
exceptions_multiclass = [44135]
correct_cate_dict = {
    53: {
        "age": False,
        "sex": True,
        "chest": True,
        "resting_blood_pressure": False,
        "serum_cholestoral": False,
        "fasting_blood_sugar": True,
        "resting_electrocardiographic_results": True,
        "maximum_heart_rate_achieved": False,
        "exercise_induced_angina": True,
        "oldpeak": False,
        "slope": True,
        "number_of_major_vessels": False,
        "thal": True,
        "class": True,
    },
}


def get_task_and_dim_out(data_id, df, columns, cate_indicator, target_col):
    """Determine Task Type and Output Dimensionality.

    Args:
        data_id (int): The OpenML dataset ID.
        df (pandas.DataFrame): The dataframe containing the dataset.
        columns (list): List of column names in the dataset.
        cate_indicator (list): List indicating whether each column is categorical or not.
        target_col (str): The name of the target column in the dataset.

    Returns:
        tuple: A tuple containing two elements - the task type ('binary', 'multiclass', 'regression')
        and the dimensionality of the output space (int).

    """

    target_idx = columns.index(target_col)

    if data_id in exceptions_binary:
        task = "binary"
        dim_out = 1
    elif data_id in exceptions_multiclass:
        task = "multiclass"
        dim_out = int(df[target_col].nunique())
    elif cont_checker(df, target_col, cate_indicator[target_idx]):
        task = "regression"
        dim_out = 1
    elif int(df[target_col].nunique()) == 2:
        task = "binary"
        dim_out = 1
    else:
        task = "multiclass"
        dim_out = int(df[target_col].nunique())
    return task, dim_out


def update_cate_indicator(data_id, cate_indicator, columns):
    """Update Categorical Indicator List.

    Args:
        data_id (int): The OpenML dataset ID.
        cate_indicator (list): List indicating whether each column is categorical or not.
        columns (list): List of column names in the dataset.

    Returns:
        list: The updated categorical indicator list reflecting the correct categorical nature
        of each column based on the predefined mapping for the given dataset.

    """

    if data_id not in correct_cate_dict:
        return cate_indicator

    for col, is_cate in zip(columns, cate_indicator):
        assert col in correct_cate_dict[data_id]
        if is_cate != correct_cate_dict[data_id][col]:
            logger.info(f"Update {col} from {is_cate} to {correct_cate_dict[data_id][col]}")
        cate_indicator[columns.index(col)] = correct_cate_dict[data_id][col]
    return cate_indicator


def cont_checker(df, col, is_cate):
    """Check if a Column is Continuous.

    Args:
        df (pandas.DataFrame): The dataframe containing the dataset.
        col (str): The name of the column to be checked.
        is_cate (bool): Indicator specifying if the column is categorical.

    Returns:
        bool: True if the column is continuous, False categorical.

    """

    return not is_cate and df[col].dtype != bool and df[col].dtype != object


def cate_checker(df, col, is_cate):
    """Check if a Column is Categorical.

    Args:
        df (pandas.DataFrame): The dataframe containing the dataset.
        col (str): The name of the column to be checked.
        is_cate (bool): Indicator specifying if the column is categorical.

    Returns:
        bool: True if the column is categorical, False continuous.

    """

    return is_cate or df[col].dtype == bool or df[col].dtype == object


def get_columns_list(df, columns, cate_indicator, target_col, checker):
    """Get List of Columns Based on Checker Function.

    Args:
        df (pandas.DataFrame): The dataframe containing the dataset.
        columns (list): List of column names in the dataset.
        cate_indicator (list): List indicating whether each column is categorical or not.
        target_col (str): The name of the target column in the dataset.
        checker (function): The function to determine if categorical or continuous.

    Returns:
        list: A filtered list of column names satisfying the conditions specified by the checker function.

    """

    return [col for col, is_cate in zip(columns, cate_indicator) if col != target_col and checker(df, col, is_cate)]


def print_dataset_details(dataset: openml.datasets.OpenMLDataset):
    """Print Dataset Details.

    Args:
        dataset (openml.datasets.OpenMLDataset): The OpenML dataset object.

    Returns:
        None

    """

    df, _, cate_indicator, columns = dataset.get_data(dataset_format="dataframe")
    print(dataset.name)
    print(dataset.openml_url)
    print(df)

    target_col = dataset.default_target_attribute
    print("Nan count", df.isna().sum().sum())
    print("cont", get_columns_list(df, columns, cate_indicator, target_col, cont_checker))
    print("cate", get_columns_list(df, columns, cate_indicator, target_col, cate_checker))
    print("target", target_col)

    task, dim_out = get_task_and_dim_out(dataset.id, df, columns, cate_indicator, target_col)
    print(f"task: {task}")
    print(f"dim_out: {dim_out}")
    print(df[target_col].value_counts())
    exit()


class OpenMLDataFrame(TabularDataFrame):
    """OpenMLDataFrame Class

    A custom class for datasets retrieved from OpenML as dataframes, designed for tabular data analysis tasks.
    This class extends the TabularDataFrame class.

    Attributes:
        continuous_columns (list): List of column names representing continuous features in the dataset.
        categorical_columns (list): List of column names representing categorical features in the dataset.
        task (str): The type of machine learning task associated with the dataset ('binary', 'multiclass', 'regression').
        dim_out (int): The dimensionality of the output space, applicable to 'multiclass' tasks.
        target_column (str): The name of the target column in the dataset.
        data (pandas.DataFrame): The cleaned The dataframe containing the dataset.

    """

    def __init__(self, id: str, show_details: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)

        dataset = openml.datasets.get_dataset(id)
        if show_details:
            print_dataset_details(dataset)

        data, _, cate_indicator, columns = dataset.get_data(dataset_format="dataframe")

        cate_indicator = update_cate_indicator(id, cate_indicator, columns)

        target_col = dataset.default_target_attribute
        self.continuous_columns = get_columns_list(data, columns, cate_indicator, target_col, cont_checker)
        self.categorical_columns = get_columns_list(data, columns, cate_indicator, target_col, cate_checker)

        self.task, self.dim_out = get_task_and_dim_out(dataset.id, data, columns, cate_indicator, target_col)

        assert self.task != "regression"

        self.target_column = target_col
        self.data = data.dropna(axis=0)
