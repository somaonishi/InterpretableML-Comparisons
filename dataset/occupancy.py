import os

import pandas as pd
from hydra.utils import to_absolute_path

from .tabular_dataframe import TabularDataFrame


class Occupancy(TabularDataFrame):
    """A class representing occupancy data in a tabular format.

    Attributes:
        data (pd.DataFrame): The DataFrame containing the occupancy data.
        target_column (str): The name of the target column representing occupancy.
        categorical_columns (list): List of column names representing categorical features in the dataset.
        continuous_columns (list): List of column names representing continuous features in the dataset.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data = pd.read_csv(to_absolute_path(os.path.join(self.root, "occupancy/datatraining.txt")), sep=",")
        self.data.drop(columns=["date"], inplace=True)
        self.target_column = "Occupancy"
        self.categorical_columns = []
        self.continuous_columns = [x for x in self.data.columns if x not in self.target_column]
