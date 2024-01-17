"""Utility Module

This module contains several functions that are used in various stages of the process

"""
import json
import operator as op
import os
import pickle
import random
from typing import Dict, Union

import numpy as np
import tensorflow as tf
from dataset.utils import feature_name_restorer
from fbts import FBT
from rerx.rule import RuleSet
from rerx.rule.rule_extraction import DecisionTreeRuleExtractor
from sklearn.metrics import accuracy_score, roc_auc_score

RANDOM_SEED = 1


def set_seed(seed: int = 42):
    """Set random seeds for TensorFlow, Python, and NumPy.

    Args:
        seed (int): The seed value to be used for random number generation. Default is 42.

    Returns:
        None

    """

    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(data: Dict[str, Union[int, float, str]], save_dir: str = "./"):
    """Save a dictionary as a JSON file.

    Args:
        data (Dict[str, Union[int, float, str]]): The dictionary to be saved as JSON.
        save_dir (str): The directory where the JSON file will be saved. Default is "./".

    Returns:
        None

    """

    with open(os.path.join(save_dir, "results.json"), mode="wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path) -> Dict[str, Union[int, float, str]]:
    """Load a JSON file and return its content as a dictionary.

    Args:
        path: The path to the JSON file.

    Returns:
        Dict[str, Union[int, float, str]]: The loaded dictionary from the JSON file.

    """

    with open(path, mode="rt", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_object(obj, output_path: str):
    """Save a object using pickle.

    Args:
        obj: The object to be saved.
        output_path (str): The path to save the object.

    Returns:
        None

    """

    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_object(input_path: str):
    """load a object from a pickle file.

    Args:
        input_path (str): The path to the pickle file.

    Returns:
        The loaded object.

    """

    with open(input_path, "rb") as f:
        return pickle.load(f)


def save_ruleset(ruleset: RuleSet, save_dir: str, file_name: str):
    """Save a RuleSet object as a pickle file and its feature names as a text file.

    Args:
        ruleset (RuleSet): The RuleSet object to be saved.
        save_dir (str): The directory where the files will be saved.
        file_name (str): The base name for the saved files.

    Returns:
        None

    """

    os.makedirs(save_dir, exist_ok=True)
    save_object(ruleset, os.path.join(save_dir, file_name) + ".pkl")
    with open(os.path.join(save_dir, file_name) + ".txt", "w") as f:
        f.write(feature_name_restorer(str(ruleset)))


def cal_auc_score(model, data, feature_cols, label_col):
    """Calculate AUC score for a given model and dataset.

    Args:
        model: The predictive model.
        data: The dataset containing feature columns and a label column.
        feature_cols: List of feature columns used for prediction.
        label_col: Name of the label column.

    Returns:
        float: The calculated AUC score.

    """

    pred_proba = model.predict_proba(data[feature_cols])
    if data[label_col].nunique() == 2:
        auc = roc_auc_score(data[label_col].values.tolist(), pred_proba[:, 1])
    else:
        auc = roc_auc_score(data[label_col].values.tolist(), pred_proba, multi_class="ovr")
    return auc


def cal_acc_score(model, data, feature_cols, label_col):
    """Calculate the accuracy score for a given model and dataset.

    Args:
        model: The predictive model.
        data: The dataset containing feature columns and a label column.
        feature_cols: List of feature columns used for prediction.
        label_col: Name of the label column.

    Returns:
        float: The calculated accuracy score.

    """

    pred = model.predict(data[feature_cols])
    acc = accuracy_score(data[label_col], pred)
    return acc


def cal_metrics(model, data, feature_cols, label_col):
    """Calculate both accuracy and AUC scores for a given model and dataset.

    Args:
        model: The predictive model.
        data: The dataset containing feature columns and a label column.
        feature_cols: List of feature columns used for prediction.
        label_col: Name of the label column.

    Returns:
        Dict[str, float]: A dictionary containing 'ACC' (accuracy) and 'AUC' scores.

    """

    acc = cal_acc_score(model, data, feature_cols, label_col)
    auc = cal_auc_score(model, data, feature_cols, label_col)
    return {"ACC": acc, "AUC": auc}


def crep(ruleset: RuleSet, X, is_decision_list=False):
    """Calculate the Complexity of Rule with Empirical Probability (CREP) for a given RuleSet and dataset.

    Args:
        ruleset (RuleSet): The RuleSet for which CREP is calculated.
        X: The dataset for which CREP is calculated.
        is_decision_list (bool): Flag indicating whether the ruleset is a decision list. Default is False.

    Returns:
        float: The calculated CREP score.

    """

    crep = 0
    conditions = 0
    covered_mask = np.zeros((X.shape[0],), dtype=bool)  # records the records that are
    for rule in ruleset:
        _, r_mask = rule.predict(X)
        # update the covered_mask with the records covered by this rule
        remaining_covered_mask = ~covered_mask & r_mask
        p_rule = remaining_covered_mask.sum() / len(X)
        if is_decision_list:
            conditions += len(rule.A)
        else:
            conditions = len(rule.A)
        crep += p_rule * conditions
        covered_mask = covered_mask | r_mask

    return crep


def set_categories_in_rule(ruleset: RuleSet, categories_dict):
    """Set categories in a RuleSet based on a dictionary.

    Args:
        ruleset (RuleSet): The RuleSet for which categories are to be set.
        categories_dict: A dictionary containing categorical column names as keys and their categories as values.

    Returns:
        None

    """

    ruleset.set_categories(categories_dict)


def softmax(x):
    """Compute the softmax function for a given array.

    Args:
        x: Input array.

    Returns:
        np.ndarray: Array of softmax values.

    """

    return np.array([np.exp(x) / np.sum(np.exp(x))])


class FBTsRuleExtractor(DecisionTreeRuleExtractor):
    """Custom rule extractor for Forest-based Trees (FBT)."""

    class Tree:
        feature_names_in_ = None

    def __init__(self, fbts: FBT, _column_names, classes_, X, y, float_threshold):
        _tree = self.Tree()
        _tree.feature_names_in_ = _column_names
        super().__init__(_tree, _column_names, classes_, X, y, float_threshold)
        self.fbts = fbts

    def get_tree_dict(self, base_tree, n_nodes=0):
        """Get a dictionary representation of the FBT tree.

        Returns:
            Dict[str, List[Union[int, float, np.ndarray, None]]]: Dictionary containing FBT tree information.

        """

        self.node_idx = -1
        keys = [
            "children_left",
            "children_right",
            "feature",
            "threshold",
            "value",
            "n_samples",
            "n_nodes",
        ]
        self.tree_dict = {k: [] for k in keys}
        self._convert(self.fbts.tree)
        self.tree_dict["n_nodes"] = self.node_idx
        return self.tree_dict

    def get_split_operators(self):
        """Get split operators for decision tree nodes.

        Returns:
            Tuple[Callable, Callable]: Tuple containing split operators.

        """

        op_left = op.ge
        op_right = op.lt
        return op_left, op_right

    def _convert(self, tree):
        """Recursively convert FBT tree nodes into dictionary format.

        Args:
            tree: Current node in the FBT tree.

        Returns:
            None

        """

        self.node_idx += 1
        feature = tree.selected_feature

        if feature is None:
            self.tree_dict["children_left"].append(-1)
            self.tree_dict["children_right"].append(-1)
            self.tree_dict["feature"].append(-2)
            self.tree_dict["threshold"].append(-2)
            self.tree_dict["value"].append(np.array([softmax(c.label_probas) for c in tree.conjunctions]).mean(0))
            self.tree_dict["n_samples"].append(None)
        else:
            this_node = self.node_idx
            right = tree.right
            left = tree.left
            threshold = tree.selected_value
            self.tree_dict["feature"].append(feature)
            self.tree_dict["threshold"].append(threshold)
            self.tree_dict["value"].append(None)
            self.tree_dict["n_samples"].append(None)
            self.tree_dict["children_left"].append(self.node_idx + 1)
            self.tree_dict["children_right"].append(None)
            self._convert(left)
            self.tree_dict["children_right"][this_node] = self.node_idx + 1
            self._convert(right)
