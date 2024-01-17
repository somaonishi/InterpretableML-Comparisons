"""Utility Module

This module provides utility functions for working with tabular data.

"""

import numpy as np
import pandas as pd


def feature_name_combiner(col, value) -> str:
    """Combines a feature name and its corresponding value into a formatted string.

    Args:
        col (str): The feature name.
        value: The value associated with the feature.

    Returns:
        str: Formatted string representing the combined feature name and value.
            Special characters in the feature name are replaced with their encoded counterparts.

    """

    def replace(s: str):
        """Helper function to replace special characters in a string."""
        return s.replace("<", "lt_").replace(">", "gt_").replace("=", "eq_").replace("[", "lb_").replace("]", "ub_")

    col = replace(str(col))
    value = replace(str(value))
    return f'{col}="{value}"'


def feature_name_restorer(feature_name: str) -> str:
    """Restores a feature name from its encoded representation.

    Args:
        feature_name (str): The encoded feature name.

    Returns:
        str: Decoded feature name with special characters restored to their original form.

    """

    return (
        feature_name.replace("lt_", "<").replace("gt_", ">").replace("eq_", "=").replace("lb_", "[").replace("ub_", "]")
    )


def label_encode(y: pd.Series):
    """Performs label encoding on a Pandas Series.

    Args:
        y (pd.Series): The targets to be encoded.

    Returns:
        pd.Series: Encoded labels represented as integers.

    """

    value_counts = y.value_counts(normalize=True)
    label_mapping = {value: index for index, (value, _) in enumerate(value_counts.items())}
    y_labels = y.map(label_mapping).astype(np.int32)
    return y_labels
