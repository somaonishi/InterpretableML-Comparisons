"""Module containing various interpretable machine learning classifiers.

Classes:
    BaseClassifier: Base class for interpretable classifiers.
    ReRxClassifier: Classifier using the Re-Rx algorithm.
    J48graftClassifier: Classifier using the J48graft decision tree algorithm.
    DTClassifier: Classifier using a Decision Tree algorithm.
    FBTsClassifier: Classifier using the FBTs algorithm.
    RuleCOSIClassifier: Classifier using the RuleCOSI algorithm.
    XGBoostClassifier: Classifier using the XGBoost algorithm.

"""

from copy import deepcopy
from typing import List

import numpy as np
import rulecosi
import rulecosi.rule_extraction as rulecosi_rule_extraction
import xgboost as xgb
from fbts import FBT
from rerx import MLP, J48graft, ReRx
from rerx.rule import RuleExtractorFactory, RuleSet
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y

from .utils import FBTsRuleExtractor, crep, set_seed


def compute_emsemble_interpretability_measures(rulesets: List[RuleSet]):
    """Compute interpretability measures for an ensemble of rule sets.

    Args:
        rulesets (List[RuleSet]): List of RuleSet instances representing the ensemble.

    Returns:
        Tuple[int, int, int]: A tuple containing the following interpretability measures:

    """

    num_rules, condition_maps, n_total_ants = 0, {}, 0
    for ruleset in rulesets:
        num_rule, _, n_total_ant = ruleset.compute_interpretability_measures()
        num_rules += num_rule
        condition_maps.update(ruleset.condition_map)
        n_total_ants += n_total_ant
    return num_rules, len(condition_maps), n_total_ants


class BaseClassifier:
    """Base class for interpretable classifiers.

    Attributes:
        ruleset (RuleSet): The rule set learned by the classifier.
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Number of output classes.
        model_config (object): Configuration for the underlying model.
        class_ratio (float): Ratio of samples in the minority class to the majority class.
        classes_ (numpy.ndarray): Unique class labels.
        onehoter (object): One-hot encoder for handling categorical features.
        verbose (int): Verbosity level for logging during training.
        pre_study (bool): Flag indicating whether a pre-study has been conducted.
        pre_model (object): Pre-trained model used for initialization.

    """

    def __init__(
        self, input_dim, output_dim, model_config, init_y, onehoter, verbose, pre_study=False, pre_model=None
    ) -> None:
        self.ruleset: RuleSet = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = model_config
        _, counts = np.unique(init_y, return_counts=True)
        self.class_ratio = counts.min() / counts.max()
        self.classes_ = unique_labels(init_y)
        self.onehoter = onehoter
        self.verbose = verbose
        self.pre_study = pre_study
        self.pre_model = pre_model

    def pre_fit(self, X, y, eval_set):
        """Perform pre-training steps before the main training.

        Args:
            X (pd.DataFrame): Training input data.
            y (numpy.ndarray): Training labels.
            eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

        Raises:
            NotImplementedError: This method should be implemented in subclasses.

        """

        raise NotImplementedError()

    def fit(self, X, y, eval_set):
        """Train the classifier on the given data.

        Args:
            X (pd.DataFrame): Training input data.
            y (numpy.ndarray): Training labels.
            eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

        Raises:
            NotImplementedError: This method should be implemented in subclasses.

        """

        raise NotImplementedError()

    def predict_proba_pre(self, X):
        """Predict class probabilities using the pre-trained model.

        Args:
            X (pd.DataFrame): Input data for which class probabilities are to be predicted.

        Returns:
            numpy.ndarray: Predicted class probabilities.

        Raises:
            NotImplementedError: This method should be implemented in subclasses if a pre-trained model is used.

        """

        return self.pre_model.predict_proba(X.values)

    def predict_pre(self, X):
        """Predict class labels using the pre-trained model.

        Args:
            X (pd.DataFrame): Input data for which class labels are to be predicted.

        Returns:
            numpy.ndarray: Predicted class labels.

        Raises:
            NotImplementedError: This method should be implemented in subclasses if a pre-trained model is used.

        """

        return self.pre_model.predict(X.values)

    def predict_proba(self, X):
        """Predict class probabilities using the learned rule set.

        Args:
            X (pd.DataFrame): Input data for which class probabilities are to be predicted.

        Returns:
            numpy.ndarray: Predicted class probabilities.

        """

        return self.ruleset.predict_proba(X.values)

    def predict(self, X):
        """Predict class labels using the learned rule set.

        Args:
            X (pd.DataFrame): Input data for which class labels are to be predicted.

        Returns:
            numpy.ndarray: Predicted class labels.

        """

        return self.ruleset.predict(X.values)

    def compute_interpretability_measures(self):
        """Compute interpretability measures for the learned rule set.

        Returns:
            Tuple[int, int, int]: A tuple containing the following interpretability measures:

        """

        return self.ruleset.compute_interpretability_measures()

    def pre_evaluate(self, X, y):
        """Evaluate performance metrics using the pre-trained model.

        Args:
            X (pd.DataFrame): Test input data.
            y (numpy.ndarray): True labels for the test data.

        Returns:
            dict: Dictionary containing evaluation results, including CREP, ACC, AUC, Precision, Recall, Specificity, and F1.

        """

        y_pred = self.predict_pre(X)
        y_score = self.predict_proba_pre(X)
        results = {}
        results["CREP"] = crep
        results["ACC"] = accuracy_score(y, y_pred)
        if self.output_dim == 2:
            results["AUC"] = roc_auc_score(y, y_score[:, 1])
            results["Precision"] = precision_score(y, y_pred, zero_division=0)
            results["Recall"] = recall_score(y, y_pred)
            results["Specificity"] = recall_score(1 - y, 1 - y_pred)
            results["F1"] = f1_score(y, y_pred, zero_division=0)
        else:
            results["AUC"] = roc_auc_score(y, y_score, multi_class="ovr")
            results["Precision"] = precision_score(y, y_pred, average="macro", zero_division=0)
            results["Recall"] = recall_score(y, y_pred, average="macro", zero_division=0)
            results["Specificity"] = recall_score(1 - y, 1 - y_pred, average="macro", zero_division=0)
            results["F1"] = f1_score(y, y_pred, average="macro", zero_division=0)
        return results

    def evaluate(self, X, y):
        """Evaluate performance metrics using the learned rule set.

        Args:
            X (pd.DataFrame): Test input data.
            y (numpy.ndarray): True labels for the test data.

        Returns:
            dict: Dictionary containing evaluation results, including Num of Rules, Ave. ante., CREP, ACC, AUC, Precision, Recall, Specificity, and F1.

        """

        y_pred = self.predict(X)
        n_rules, _, n_total_ant = self.compute_interpretability_measures()
        crep = self.compute_crep(X)
        results = {}
        results["Num of Rules"] = n_rules
        results["Ave. ante."] = n_total_ant / n_rules
        results["CREP"] = crep
        results["ACC"] = accuracy_score(y, y_pred)
        if self.output_dim == 2:
            results["AUC"] = roc_auc_score(y, y_pred)
            results["Precision"] = precision_score(y, y_pred, zero_division=0)
            results["Recall"] = recall_score(y, y_pred)
            results["Specificity"] = recall_score(1 - y, 1 - y_pred)
            results["F1"] = f1_score(y, y_pred, zero_division=0)
        else:
            y_score = self.predict_proba(X)
            results["AUC"] = roc_auc_score(y, y_score, multi_class="ovr")
            results["Precision"] = precision_score(y, y_pred, average="macro", zero_division=0)
            results["Recall"] = recall_score(y, y_pred, average="macro", zero_division=0)
            results["Specificity"] = recall_score(1 - y, 1 - y_pred, average="macro", zero_division=0)
            results["F1"] = f1_score(y, y_pred, average="macro", zero_division=0)
        return results

    def compute_crep(self, X):
        """Compute Classification Rule Evaluation Paradigm (CREP) for the learned rule set.

        Args:
            X (pd.DataFrame): Input data for which CREP is to be computed.

        Returns:
            float: CREP value for the learned rule set.

        """

        crep_result = crep(self.ruleset, X.values)
        return crep_result


class ReRxClassifier(BaseClassifier):
    """Classifier using the Re-Rx algorithm.

    This classifier combines a Multi-Layer Perceptron (MLP) base model with a J48graft decision tree
    to achieve interpretability. It uses the Re-Rx algorithm for rule extraction.

    Attributes:
        mlp (MLP): Multi-Layer Perceptron base model for the Re-Rx algorithm.
        tree (J48graft): J48graft decision tree for the Re-Rx algorithm.
        rerx (ReRx): Re-Rx algorithm combining the MLP and decision tree.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        rerx_config = self.model_config.rerx
        mlp_config = self.model_config.mlp
        tree_config = self.model_config.tree
        self.mlp = MLP(
            epochs=200,
            early_stop=True,
            use_output_bias=True,
            verbose=self.verbose,
            onehoter=self.onehoter,
            **mlp_config,
        )
        tree = J48graft(**tree_config, verbose=self.verbose)
        self.rerx = ReRx(
            base_model=self.mlp,
            tree=tree,
            output_dim=self.output_dim,
            is_eval=self.verbose > 0,
            verbose=self.verbose,
            **rerx_config,
        )

    def pre_fit(self, X, y, eval_set):
        """Perform pre-training steps before the main training.

        Args:
            X (pd.DataFrame): Training input data.
            y (numpy.ndarray): Training labels.
            eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

        """

        label_map = self.rerx.get_label_map(y)
        if label_map is not None:
            y = self.rerx.map_y(y, label_map)
            if eval_set is not None:
                eval_set = self.rerx.update_eval_set(eval_set, label_map)
        self.mlp.fit(X, y, eval_set=eval_set)
        self.pre_model = self.mlp

    def predict_pre(self, X):
        """
        Predict class labels using the pre-trained MLP model.

        Args:
            X (pd.DataFrame): Input data for which class labels are to be predicted using the pre-trained model.

        Returns:
            numpy.ndarray: Predicted class labels.

        """

        return self.pre_model.predict(X)

    def predict_proba_pre(self, X):
        """Predict class probabilities using the pre-trained MLP model.

        Args:
            X (pd.DataFrame): Input data for which class probabilities are to be predicted using the pre-trained model.

        Returns:
            numpy.ndarray: Predicted class probabilities.

        """

        return self.pre_model.predict_proba(X)

    def fit(self, X, y, eval_set):
        """Train the classifier using the Re-Rx algorithm.

        Args:
            X (pd.DataFrame): Training input data.
            y (numpy.ndarray): Training labels.
            eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

        """

        self.rerx.fit(X, y, eval_set=eval_set)
        self.ruleset = self.rerx.ruleset


class J48graftClassifier(BaseClassifier):
    """Train the classifier using the J48graft decision tree algorithm.

    This classifier employs the J48graft decision tree algorithm for rule extraction and interpretability.

    Args:
        X (pd.DataFrame): Training input data.
        y (numpy.ndarray): Training labels.
        eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tree = J48graft(**self.model_config, verbose=self.verbose)

    def fit(self, X, y, eval_set):
        """Train the classifier using the J48graft decision tree algorithm.

        Args:
            X (pd.DataFrame): Training input data.
            y (numpy.ndarray): Training labels.
            eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

        """

        self.tree.fit(X, y, eval_set=eval_set)
        extractor = RuleExtractorFactory.get_rule_extractor(
            self.tree, X.columns.to_list(), unique_labels(y), None, y, 0
        )
        self.ruleset, _ = extractor.extract_rules()


class DTClassifier(BaseClassifier):
    """Classifier using the Decision Tree algorithm.

    This classifier employs the Decision Tree algorithm for rule extraction and interpretability.

    Attributes:
        tree (DecisionTreeClassifier): Decision Tree classifier for rule extraction.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tree = DecisionTreeClassifier(**self.model_config)

    def fit(self, X, y, eval_set):
        """Train the classifier using the Decision Tree algorithm.

        Args:
            X (pd.DataFrame): Training input data.
            y (numpy.ndarray): Training labels.
            eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

        """

        self.tree.fit(X, y)
        extractor = RuleExtractorFactory.get_rule_extractor(
            self.tree, X.columns.to_list(), unique_labels(y), None, y, 0
        )
        self.ruleset, _ = extractor.extract_rules()


class FBTsClassifier(BaseClassifier):
    """Classifier using the Feature-based Tree Search (FBTs) algorithm.

    This classifier combines an XGBoost model with the FBTs algorithm for rule extraction and interpretability.

    Attributes:
        ens (xgb.XGBClassifier): XGBoost classifier used in the ensemble.
        fbts (FBT): Feature-based Tree Search (FBTs) algorithm for rule extraction.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        xgb_config = self.model_config.ensemble
        fbts_config = self.model_config.fbts

        if self.output_dim == 2:
            xgb_config["objective"] = "binary:logitraw"
        else:
            xgb_config["objective"] = "multi:softmax"

        if self.pre_model is not None:
            self.ens = self.pre_model
        else:
            self.ens = xgb.XGBClassifier(
                **xgb_config,
                num_class=self.output_dim if self.output_dim > 2 else None,
                eval_metric="auc",
                early_stopping_rounds=10,
            )
        self.fbts = FBT(**fbts_config, verbose=self.verbose)

    def pre_fit(self, X, y, eval_set):
        """Perform pre-training steps before the main training.

        This method pre-trains the XGBoost model on the given data, which will be used as part of the FBTs classifier.

        Args:
            X (pd.DataFrame): Training input data.
            y (numpy.ndarray): Training labels.
            eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

        """

        self.pre_model = self.ens.fit(X, y, eval_set=[eval_set], verbose=False)

    def fit(self, X, y, eval_set):
        """Train the classifier using the Feature-based Tree Search (FBTs) algorithm.

        This method trains the FBTs classifier by combining an XGBoost model with the FBTs algorithm for rule extraction.

        Args:
            X (pd.DataFrame): Training input data.
            y (numpy.ndarray): Training labels.
            eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

        """

        if self.pre_model is None:
            self.ens.fit(X, y, eval_set=[eval_set], verbose=False)

        data = deepcopy(X)
        columns = list(data.columns)
        data["class"] = y
        self.fbts.fit(data, columns, "class", self.ens)
        extractor = FBTsRuleExtractor(self.fbts, columns, unique_labels(y), None, y, 0)
        self.ruleset, _ = extractor.extract_rules()


class RuleCOSIClassifier(BaseClassifier):
    """Classifier using the RuleCOSI algorithm.

    This classifier combines an XGBoost model with the RuleCOSI algorithm for rule extraction and interpretability.

    Attributes:
        ens (xgb.XGBClassifier): XGBoost classifier used in the ensemble.
        rulecosi (rulecosi.RuleCOSIClassifier): RuleCOSI algorithm for rule extraction.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        xgb_config = self.model_config.ensemble
        rulecosi_config = self.model_config.rulecosi

        if self.output_dim == 2:
            xgb_config["objective"] = "binary:logitraw"
        else:
            xgb_config["objective"] = "multi:softmax"

        if self.pre_model is not None:
            self.ens = self.pre_model
        else:
            self.ens = xgb.XGBClassifier(
                **xgb_config,
                num_class=self.output_dim if self.output_dim > 2 else None,
                eval_metric="auc",
                early_stopping_rounds=10,
            )
        self.rulecosi = rulecosi.RuleCOSIClassifier(
            metric="auc",
            **rulecosi_config,
            verbose=self.verbose,
        )

    def pre_fit(self, X, y, eval_set):
        """Perform pre-training steps before the main training.

        Args:
            X (pd.DataFrame): Training input data.
            y (numpy.ndarray): Training labels.
            eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

        """

        X_xgb, y_xgb = check_X_y(X, y)
        eval_set_xgb = check_X_y(*eval_set)
        self.pre_model = self.ens.fit(X_xgb, y_xgb, eval_set=[eval_set_xgb], verbose=False)

    def fit(self, X, y, eval_set):
        """Train the classifier using the RuleCOSI algorithm.

        This method trains the RuleCOSI classifier by combining an XGBoost model with the RuleCOSI algorithm for rule extraction.

        Args:
            X (pd.DataFrame): Training input data.
            y (numpy.ndarray): Training labels.
            eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

        """

        if self.pre_model is None:
            X_xgb, y_xgb = check_X_y(X, y)
            eval_set_xgb = check_X_y(*eval_set)
            self.ens.fit(X_xgb, y_xgb, eval_set=[eval_set_xgb], verbose=False)

        self.rulecosi.fit(X, y)
        self.ruleset = self.rulecosi.simplified_ruleset_

    def predict_proba(self, X):
        """Predict class probabilities for input data using the simplified ruleset.

        This method predicts class probabilities using the simplified ruleset obtained from the RuleCOSI algorithm.

        Args:
            X (pd.DataFrame): Input data for which class probabilities are to be predicted.

        Returns:
            numpy.ndarray: Predicted class probabilities.

        """

        return softmax(self.ruleset.predict_proba(X.values), axis=1)

    def compute_crep(self, X):
        """Compute Classification Rule Evaluation Performance (CREP) for input data.

        This method computes CREP using the simplified ruleset obtained from the RuleCOSI algorithm.

        Args:
            X (pd.DataFrame): Input data for which CREP is to be computed.

        Returns:
            float: CREP value.

        """

        crep_result = crep(self.ruleset, X.values, is_decision_list=True)
        return crep_result


class XGBoostClassifier(BaseClassifier):
    """Classifier using the XGBoost algorithm.

    This classifier employs the XGBoost algorithm for rule extraction and interpretability.

    Attributes:
        xgb (xgb.XGBClassifier): XGBoost classifier for rule extraction.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.output_dim == 2:
            self.model_config["objective"] = "binary:logitraw"
        else:
            self.model_config["objective"] = "multi:softmax"

        self.xgb = xgb.XGBClassifier(
            **self.model_config,
            num_class=self.output_dim if self.output_dim > 2 else None,
            eval_metric="auc",
            early_stopping_rounds=10,
        )

    def fit(self, X, y, eval_set):
        """Train the classifier using the XGBoost algorithm.

        This method trains the XGBoost classifier on the given data.

        Args:
            X (pd.DataFrame): Training input data.
            y (numpy.ndarray): Training labels.
            eval_set (tuple): Evaluation set in the form of (X_eval, y_eval).

        """

        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = check_X_y(*eval_set)

        self.xgb.fit(X, y, eval_set=[eval_set], verbose=False)
        extractor = rulecosi_rule_extraction.RuleExtractorFactory.get_rule_extractor(
            self.xgb,
            self._column_names,
            self.classes_,
            X,
            y,
            1e-6,
        )
        self.rulesets, _ = extractor.extract_rules()

    def compute_crep(self, X):
        """Compute Classification Rule Evaluation Performance (CREP) for input data.

        This method computes CREP by aggregating CREP values from individual rulesets obtained from the XGBoost model.

        Args:
            X (pd.DataFrame): Input data for which CREP is to be computed.

        Returns:
            float: Aggregate CREP value.

        """

        crep_sum = 0
        for ruleset in self.rulesets:
            crep_sum += crep(ruleset, X.values)

        return crep_sum

    def predict_proba(self, X):
        """Predict class probabilities for input data.

        This method predicts class probabilities using the XGBoost model.

        Args:
            X (pd.DataFrame): Input data for which class probabilities are to be predicted.

        Returns:
            numpy.ndarray: Predicted class probabilities.

        """

        return self.xgb.predict_proba(X)

    def predict(self, X):
        """Predict class labels for input data.

        This method predicts class labels using the XGBoost model.

        Args:
            X (pd.DataFrame): Input data for which class labels are to be predicted.

        Returns:
            numpy.ndarray: Predicted class labels.

        """

        return self.xgb.predict(X)

    def compute_interpretability_measures(self):
        """Compute interpretability measures for the ensemble.

        This method computes interpretability measures for the ensemble of rulesets obtained from the XGBoost model.

        Returns:
            tuple: Number of rules, unique conditions, and total number of antecedents in the ensemble.

        """

        return compute_emsemble_interpretability_measures(self.rulesets)


def get_classifier(
    name,
    *,
    input_dim,
    output_dim,
    model_config,
    init_y,
    onehoter,
    pre_study=False,
    pre_model=None,
    seed=42,
    verbose=0,
):
    """Get a classifier based on the specified name.

    This function returns an instance of a classifier based on the specified name and configuration parameters.

    Args:
        name (str): Name of the classifier to be instantiated.
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output labels.
        model_config (dict): Configuration parameters for the model.
        init_y (numpy.ndarray): Initial labels for training.
        onehoter: One-hot encoder for handling categorical features.
        pre_study (bool): Whether to perform a pre-study.
        pre_model: Pre-trained model if available.
        seed (int): Seed for random number generation.
        verbose (int): Verbosity level.

    Returns:
        BaseClassifier: Instance of the specified classifier.

    """

    set_seed(seed=seed)
    if name == "rerx":
        return ReRxClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose, pre_study)
    elif name == "rulecosi":
        return RuleCOSIClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose, pre_study, pre_model)
    elif name == "fbts":
        return FBTsClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose, pre_study, pre_model)
    elif name == "xgboost":
        return XGBoostClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose)
    elif name == "j48graft":
        return J48graftClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose)
    elif name == "dt":
        return DTClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose)
    else:
        raise KeyError(f"{name} is not defined.")
