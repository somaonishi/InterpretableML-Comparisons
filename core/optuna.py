"""Optuna Module

This module provides a class, OptunaSearch, that performs hyperparameter optimization using the Optuna framework.

"""

import logging
import os
from copy import deepcopy
from statistics import mean

import numpy as np
import optuna
from hydra.utils import to_absolute_path
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn.model_selection import StratifiedKFold

from .classifier import get_classifier

logger = logging.getLogger(__name__)


def xgboost_config(trial: optuna.Trial, model_config, name=""):
    """Configures hyperparameters for the XGBoost.

    Args:
        trial (optuna.Trial): The Optuna trial object used for hyperparameter optimization.
        model_config (object): The configuration object for the XGBoost.
        name (str): Prefix for the hyperparameter names.

    Returns:
        object: Updated XGBoost configuration.

    """

    model_config.max_depth = trial.suggest_int(f"{name}max_depth", 1, 10)
    model_config.eta = trial.suggest_float(f"{name}eta", 1e-4, 1.0, log=True)
    model_config.n_estimators = 250
    return model_config


def j48graft_config(trial: optuna.Trial, model_config, name=""):
    """Configures hyperparameters for the J48Graft.

    Args:
        trial (optuna.Trial): The Optuna trial object used for hyperparameter optimization.
        model_config (object): The configuration object for the J48Graft.
        name (str): Prefix for the hyperparameter names.

    Returns:
        object: Updated J48Graft configuration.

    """

    min_instance_power = trial.suggest_int(f"{name}min_instance_power", 1, 8)
    model_config.min_instance = 2**min_instance_power
    model_config.pruning_conf = trial.suggest_float(f"{name}pruning_conf", 0.1, 0.5)
    return model_config


def rerx_config(trial: optuna.Trial, model_config, pre_study=False):
    """Configures hyperparameters for the ReRx.

    Args:
        trial (optuna.Trial): The Optuna trial object used for hyperparameter optimization.
        model_config (object): The configuration object for the ReRx.
        pre_study (bool): Indicates if the configuration is for pre-study optimization.

    Returns:
        object: Updated ReRx configuration.

    """

    if pre_study:
        model_config.mlp.h_dim = trial.suggest_int("mlp.h_dim", 1, 5)
        model_config.mlp.lr = trial.suggest_float("mlp.lr", 5e-3, 0.1, log=True)
        model_config.mlp.lr = trial.suggest_float("mlp.weight_decay", 1e-6, 1e-2, log=True)
    else:
        model_config.tree = j48graft_config(trial, model_config.tree, name="tree.")
        model_config.rerx.pruning_lamda = trial.suggest_float("rerx.pruning_lamda", 0.001, 0.25, log=True)
        model_config.rerx.delta_1 = trial.suggest_float("rerx.delta_1", 0.05, 0.4)
        model_config.rerx.delta_2 = trial.suggest_float("rerx.delta_2", 0.05, 0.4)
    return model_config


def rulecosi_config(trial: optuna.Trial, model_config, pre_study=False):
    """Configures hyperparameters for the RuleCOSI.

    Args:
        trial (optuna.Trial): The Optuna trial object used for hyperparameter optimization.
        model_config (object): The configuration object for the RuleCOSI.
        pre_study (bool): Indicates if the configuration is for pre-study optimization.

    Returns:
        object: Updated RuleCOSI configuration.

    """

    if pre_study:
        model_config.ensemble = xgboost_config(trial, model_config.ensemble, name="ensemble.")
    else:
        model_config.rulecosi.conf_threshold = trial.suggest_float("rulecosi.conf_threshold", 0.0, 0.95)
        model_config.rulecosi.cov_threshold = trial.suggest_float("rulecosi.cov_threshold", 0.0, 0.5)
        model_config.rulecosi.c = trial.suggest_float("rulecosi.c", 0.1, 0.5)
    return model_config


def fbts_config(trial: optuna.Trial, model_config, pre_study=False):
    """Configures hyperparameters for the FBTs.

    Args:
        trial (optuna.Trial): The Optuna trial object used for hyperparameter optimization.
        model_config (object): The configuration object for the FBTs.
        pre_study (bool): Indicates if the configuration is for pre-study optimization.

    Returns:
        object: Updated FBTs configuration.

    """

    if pre_study:
        model_config.ensemble = xgboost_config(trial, model_config.ensemble, name="ensemble.")
    else:
        model_config.fbts.max_depth = trial.suggest_int("fbts.max_depth", 1, 10)
        model_config.fbts.pruning_method = trial.suggest_categorical("fbts.pruning_method", [None, "auc"])
    return model_config


def dt_config(trial: optuna.Trial, model_config):
    """Configures hyperparameters for the Decision Tree.

    Args:
        trial (optuna.Trial): The Optuna trial object used for hyperparameter optimization.
        model_config (object): The configuration object for the Decision Tree.

    Returns:
        object: Updated Decision Tree configuration.

    """

    model_config.max_depth = trial.suggest_int("max_depth", 1, 10)
    model_config.min_samples_split = trial.suggest_float("min_samples_split", 0.0, 0.5)
    model_config.min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.0, 0.5)
    return model_config


def get_model_config(model_name):
    """Returns the hyperparameter configuration function for the specified machine learning model.

    Args:
        model_name (str): The name of the machine learning model.

    Returns:
        function: Hyperparameter configuration function for the specified model.

    Raises:
        ValueError: If the specified model name is not supported.

    """

    if model_name == "rerx":
        return rerx_config
    elif model_name == "rulecosi":
        return rulecosi_config
    elif model_name == "fbts":
        return fbts_config
    elif model_name == "xgboost":
        return xgboost_config
    elif model_name == "j48graft":
        return j48graft_config
    elif model_name == "dt":
        return dt_config
    else:
        raise ValueError()


def update_model_cofig(default_config, best_config):
    """Updates the default hyperparameter configuration with the best configuration.

    Args:
        default_config (dict): The default hyperparameter configuration.
        best_config (dict): The best hyperparameter configuration obtained from Optuna.

    """

    for _p, v in best_config.items():
        current_dict = default_config
        _p = _p.split(".")
        for p in _p[:-1]:
            if p not in current_dict:
                current_dict[p] = {}
            current_dict = current_dict[p]
        last_key = _p[-1]
        current_dict[last_key] = v


class OptunaSearch:
    """Hyperparameter optimization using Optuna for machine learning models."""

    def __init__(
        self,
        model_name,
        default_config,
        input_dim,
        output_dim,
        X,
        y,
        val_data,
        columns,
        target_column,
        onehoter,
        n_trials,
        n_startup_trials,
        storage,
        study_name,
        seed=42,
        alpha=1,
    ) -> None:
        self.model_name = model_name
        self.default_config = deepcopy(default_config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = get_model_config(model_name)
        self.X = X
        self.y = y
        self.val_data = val_data
        self.columns = columns
        self.target_column = target_column
        self.onehoter = onehoter
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.storage = to_absolute_path(storage) if storage is not None else None
        self.study_name = study_name
        self.seed = seed
        self.alpha = alpha
        self.pre_study = False
        if "rerx" in self.model_name or "rulecosi" in self.model_name or "fbts" in self.model_name:
            self.pre_study = True
            self.n_trials = self.n_trials // 2

    def fit(self, model_config, X_train, y_train, X_val=None, y_val=None):
        """Fits the model using the given hyperparameter configuration and training data.

        Args:
            model_config (object): The hyperparameter configuration for the model.
            X_train (pd.DataFrame): The input training data.
            y_train (pd.Series): The target training labels.
            X_val (pd.DataFrame): Validation data for early stopping. Default is None.
            y_val (pd.Series): Validation labels. Default is None.

        Returns:
            dict: Evaluation score of the fitted model.

        """

        if X_val is None and y_val is None:
            X_val = self.val_data[self.columns]
            y_val = self.val_data[self.target_column].values.squeeze()

        model = get_classifier(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
            init_y=self.y,
            onehoter=self.onehoter,
            seed=self.seed,
            pre_study=self.pre_study,
            pre_model=deepcopy(self.pre_model) if hasattr(self, "pre_model") else None,
        )
        fit = model.pre_fit if self.pre_study else model.fit
        fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
        )
        evaluate = model.pre_evaluate if self.pre_study else model.evaluate
        score = evaluate(
            self.val_data[self.columns],
            self.val_data[self.target_column].values.squeeze(),
        )
        return score

    def fit_pre_model(self, model_config, X_train, y_train, X_val=None, y_val=None):
        """Fits a pre-model used for models like RERX, RuleCOSI, and FBTS.

        This method is specifically designed for models that require a pre-training step before the
        main optimization. Examples include RERX (Rule Ensemble with Rule Execution), RuleCOSI (Rule-based
        Cost-sensitive learning), and FBTS (Feature-Based Time Series). The pre-model is trained on the
        provided training data, and the resulting pre-trained model is stored in the instance variable
        `pre_model`.

        Args:
            model_config (object): The hyperparameter configuration for the pre-model.
            X_train (pd.DataFrame): The input training data.
            y_train (pd.Series): The target training labels.
            X_val (pd.DataFrame): Validation data for early stopping. Defaults to None.
            y_val (pd.Series): Validation labels. Defaults to None.

        Returns:
            None

        """

        logger.info("Fitting pre model...")
        if X_val is None and y_val is None:
            X_val = self.val_data[self.columns]
            y_val = self.val_data[self.target_column].values.squeeze()

        model = get_classifier(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
            init_y=self.y,
            onehoter=self.onehoter,
            seed=self.seed,
            pre_study=True,
        )
        model.pre_fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
        )
        self.pre_model = model.pre_model

    def get_score(self, model_config):
        """Computes and returns the evaluation score using the specified hyperparameter configuration.

        Args:
            model_config (object): The configuration for the model.

        Returns:
            tuple: A tuple containing the AUC score and the inverse of the number of rules.

        """

        score = self.fit(model_config, self.X, self.y)
        return score["AUC"], 1 / score["Num of Rules"]

    def pre_objective(self, trial):
        """Objective function for pre-study optimization.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: AUC score obtained during pre-study optimization.

        """

        _model_config = self.model_config(trial, deepcopy(self.default_config), pre_study=True)
        score = self.fit(_model_config, self.X, self.y)
        return score["AUC"]

    def objective(self, trial):
        """Objective function for the main study optimization.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            tuple: A tuple containing the AUC score and the inverse of the number of rules.

        """

        _model_config = self.model_config(trial, deepcopy(self.default_config))
        auc, inv_rules = self.get_score(_model_config)
        return auc, inv_rules

    def _get_best_params(self, study: optuna.Study):
        """Performs hyperparameter optimization and returns the best configuration for the specified model.

        Returns:
            dict: The best hyperparameter configuration.

        """

        best_trials = study.best_trials
        best_trials = [trial for trial in best_trials if trial.values[0] != 0.5]
        if len(best_trials) == 0:
            best_trials = [study.best_trials[0]]
        best_trial = best_trials[0]
        k_best = -np.inf
        for trial in best_trials:
            x, y = trial.values
            k = np.log2(y) + 100 * self.alpha * x
            # print(f"{k:.3f}, {y:.3f}, {np.log2(y):.1f}, {x:.3f}")
            if k > k_best:
                k_best = k
                best_trial = trial
        logger.info(f"Accepted trial: {best_trial.number}")
        logger.info(f"AUC: {best_trial.values[0]}, 1/Rules: {best_trial.values[1]}, Rules: {1/best_trial.values[1]}")
        logger.info(f"Parameters: {best_trial.params}")
        return best_trial.params

    def get_n_complete(self, study: optuna.Study):
        """Gets the number of completed trials in the given Optuna study.

        Args:
            study (optuna.Study): The Optuna study object.

        Returns:
            int: The number of completed trials.

        """

        n_complete = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
        return n_complete

    def get_best_config(self):
        """Performs hyperparameter optimization and returns the best configuration for the specified model.

        This method orchestrates the hyperparameter optimization process. It creates an Optuna study, performs
        optimization, and retrieves the best hyperparameter configuration. The optimization is conducted separately
        for the pre-model (if applicable) and the main model. The resulting best hyperparameters are then used to
        update the default configuration.

        Returns:
            dict: The best hyperparameter configuration.

        """

        if self.storage is not None:
            os.makedirs(self.storage, exist_ok=True)
            self.storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{self.storage}/optuna.db",
            )
        if "rerx" in self.model_name or "rulecosi" in self.model_name or "fbts" in self.model_name:
            study_name = f"{self.study_name}_pre"
            if "fbts" in study_name or "rulecosi" in study_name:
                study_name = study_name.replace(self.model_name, "xgboost")

            pre_study = optuna.create_study(
                storage=self.storage,
                study_name=study_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=self.seed,
                    n_startup_trials=self.n_startup_trials,
                ),
                load_if_exists=True,
            )
            n_complete = self.get_n_complete(pre_study)
            if self.n_trials > n_complete:
                pre_study.optimize(
                    self.pre_objective,
                    n_jobs=1,
                    callbacks=[MaxTrialsCallback(self.n_trials, states=(TrialState.COMPLETE,))],
                )
            best_params = pre_study.best_params
            update_model_cofig(self.default_config, best_params)
            self.pre_study = False
            if self.model_name != "rerx":
                self.fit_pre_model(self.default_config, self.X, self.y)

        study = optuna.create_study(
            storage=self.storage,
            study_name=self.study_name,
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.TPESampler(
                seed=self.seed,
                n_startup_trials=self.n_startup_trials,
            ),
            load_if_exists=True,
        )
        n_complete = self.get_n_complete(study)
        if self.n_trials > n_complete:
            study.optimize(
                self.objective,
                n_jobs=1,
                callbacks=[MaxTrialsCallback(self.n_trials, states=(TrialState.COMPLETE,))],
            )
        best_params = self._get_best_params(study)
        if "j48graft" in self.model_name:
            best_params["min_instance"] = 2 ** best_params["min_instance_power"]
            del best_params["min_instance_power"]
        if "rerx" in self.model_name:
            best_params["tree.min_instance"] = 2 ** best_params["tree.min_instance_power"]
            del best_params["tree.min_instance_power"]
        update_model_cofig(self.default_config, best_params)
        return self.default_config
