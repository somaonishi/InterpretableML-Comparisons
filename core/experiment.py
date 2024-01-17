import logging
import os
from time import time

import numpy as np
import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import dataset
from dataset import TabularDataFrame

from .classifier import get_classifier
from .optuna import OptunaSearch
from .utils import (
    cal_metrics,
    load_json,
    save_json,
    save_ruleset,
    set_categories_in_rule,
    set_seed,
)

logger = logging.getLogger(__name__)


class ExpBase:
    """Base class for experimental configurations and procedures.

    Attributes:
        n_splits (int): Number of splits for cross-validation.
        model_name (str): Name of the model used in the experiment.
        model_config (dict): Configuration parameters for the model.
        exp_config (Config): Experiment configuration.
        data_config (Config): Data configuration.
        categories_dict (dict): Dictionary of categorical columns and their categories.
        train (DataFrame): Training data.
        test (DataFrame): Test data.
        columns (List[str]): List of all column names in the dataset.
        target_column (str): Name of the target column.
        onehoter (OneHotEncoder or None): OneHotEncoder for MLP in ReRx, None for other models.
        seed (int): Random seed for reproducibility.
        writer (dict): Dictionary to store experiment results.

    """

    def __init__(self, config):
        set_seed(config.seed)

        self.n_splits = config.n_splits
        self.model_name = config.model.name

        self.model_config = config.model.params
        self.exp_config = config.exp
        self.data_config = config.data

        dataframe: TabularDataFrame = getattr(dataset, self.data_config.name)(seed=config.seed, **self.data_config)
        dfs = dataframe.processed_dataframes()
        self.categories_dict = dataframe.get_categories_dict()
        self.train, self.test = dfs["train"], dfs["test"]
        self.columns = dataframe.all_columns
        self.target_column = dataframe.target_column

        # Onehotter for MLP in ReRx
        if self.model_name == "rerx":
            all_cate = pd.concat([self.train, self.test])[dataframe.categorical_columns]
            self.onehoter = OneHotEncoder(sparse_output=False).fit(all_cate) if len(all_cate.columns) != 0 else None
        else:
            self.onehoter = None

        self.seed = config.seed
        self.init_writer()

    def init_writer(self):
        """Initialize the result writer.

        This method sets up the initial structure for storing experiment results.

        """

        metrics = [
            "fold",
            "ACC",
            "AUC",
            "Num of Rules",
            "Ave. ante.",
            "CREP",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "Time",
        ]
        self.writer = {m: [] for m in metrics}
        if os.path.exists("results.json"):
            _writer = load_json("results.json")
            for k, v in _writer.items():
                self.writer[k] = v
            self.writer["fold"] = list(range(len(_writer["Time"])))

    def add_results(self, i_fold, scores: dict, time):
        """Add results for a specific fold to the result writer.

        This method appends evaluation results for a specific fold to the result writer.
        It includes metrics such as accuracy (ACC), area under the curve (AUC),
        number of rules (Num of Rules), average anticipation (Ave. ante.), CREP,
        precision, recall, specificity, F1 score, and execution time.

        Args:
            i_fold (int): Fold index.
            scores (dict): Dictionary containing evaluation metrics.
            time (float): Execution time.

        """

        self.writer["fold"].append(i_fold)
        self.writer["Time"].append(time)
        for m, score in scores.items():
            self.writer[m].append(score)
        save_json({k: v for k, v in self.writer.items() if k != "fold"})

    def each_fold(self, i_fold, train_data, val_data):
        """Execute the experiment for each fold.

        This method performs the experiment for a specific fold, including training
        the model on the training data, evaluating it on the validation data, and
        logging the results.

        Args:
            i_fold (int): Fold index.
            train_data (DataFrame): Training data for the current fold.
            val_data (DataFrame): Validation data for the current fold.

        Returns:
            tuple: A tuple containing the trained model and execution time.

        """

        uniq = self.get_unique(train_data)
        x, y = self.get_x_y(train_data)

        model_config = self.get_model_config(i_fold=i_fold, x=x, y=y, val_data=val_data, uniq=uniq)
        model = get_classifier(
            self.model_name,
            input_dim=len(self.columns),
            output_dim=len(uniq),
            model_config=model_config,
            init_y=y,
            onehoter=self.onehoter,
            verbose=self.exp_config.verbose,
            seed=self.seed,
        )
        start = time()
        model.fit(
            x,
            y,
            eval_set=(val_data[self.columns], val_data[self.target_column].values.squeeze()),
        )
        end = time() - start
        logger.info(f"[Fit {self.model_name}] Time: {end}")
        return model, end

    def run(self):
        """Run the experiment.

        This method orchestrates the execution of the entire experiment, including
        the cross-validation process, training and evaluation for each fold,
        logging results, and saving the ruleset.

        """

        skf = StratifiedKFold(n_splits=self.n_splits)
        for i_fold, (train_idx, val_idx) in enumerate(skf.split(self.train, self.train[self.target_column])):
            if len(self.writer["fold"]) != 0 and self.writer["fold"][-1] >= i_fold:
                logger.info(f"Skip {i_fold + 1} fold. Already finished.")
                continue

            train_data, val_data = self.train.iloc[train_idx], self.train.iloc[val_idx]
            model, time = self.each_fold(i_fold, train_data, val_data)

            score = cal_metrics(model, val_data, self.columns, self.target_column)
            score.update(model.evaluate(val_data[self.columns], val_data[self.target_column].values.squeeze()))
            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] val/ACC: {score['ACC']:.4f} | val/AUC: {score['AUC']:.4f} | "
                f"val/Rules: {score['Num of Rules']}"
            )

            score = cal_metrics(model, self.test, self.columns, self.target_column)
            score.update(model.evaluate(self.test[self.columns], self.test[self.target_column].values.squeeze()))

            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] test/ACC: {score['ACC']:.4f} | test/AUC: {score['AUC']:.4f} | "
                f"test/Rules: {score['Num of Rules']}"
            )
            self.add_results(i_fold, score, time)

            if self.model_name == "rerx" and self.categories_dict is not None:
                set_categories_in_rule(model.ruleset, self.categories_dict)

            save_ruleset(model.ruleset, save_dir="ruleset", file_name=f"ruleset_{i_fold+1}")

        logger.info(f"[{self.model_name} Test Results]")
        mean_std_score = {}
        score_list_dict = {}
        for k, score_list in self.writer.items():
            if k == "fold":
                continue
            score = np.array(score_list)
            mean_std_score[k] = f"{score.mean(): .4f} ±{score.std(ddof=1): .4f}"
            score_list_dict[k] = score_list
            logger.info(f"[{self.model_name} {k}]: {mean_std_score[k]}")
        save_json(score_list_dict)

    def get_model_config(self, *args, **kwargs):
        """Get the model configuration for each fold.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.

        """

        raise NotImplementedError()

    def get_unique(self, train_data):
        """Get unique values of the target column in the training data.

        Args:
            train_data (DataFrame): Training data.

        Returns:
            ndarray: Unique values of the target column.

        """

        uniq = np.unique(train_data[self.target_column])
        return uniq

    def get_x_y(self, train_data):
        """Get input features and target values from the training data.

        Args:
            train_data (DataFrame): Training data.

        Returns:
            tuple: Input features and target values.

        """

        x, y = train_data[self.columns], train_data[self.target_column].values.squeeze()
        return x, y


class ExpSimple(ExpBase):
    """Simple experimental configuration without hyperparameter tuning.

    This class inherits from ExpBase and is designed for experiments without
    hyperparameter tuning. It provides a straightforward configuration where the
    model is trained and evaluated using the specified parameters.

    Attributes:
        Inherits attributes from ExpBase.

    """

    def __init__(self, config):
        super().__init__(config)

    def get_model_config(self, *args, **kwargs):
        """Get the model configuration for each fold.

        Returns:
            dict: Model configuration.

        """
        return self.model_config


class ExpOptuna(ExpBase):
    """Experimental configuration with hyperparameter tuning using Optuna.

    This class inherits from ExpBase and is designed for experiments with
    hyperparameter tuning using the Optuna library. It includes methods to run
    experiments with Optuna optimization and optionally delete Optuna study data.

    Attributes:
        Inherits attributes from ExpBase.
        n_trials (int): Number of trials for Optuna optimization.
        n_startup_trials (int): Number of initial trials for Optuna optimization.
        storage (str): Storage URL for Optuna study data.
        study_name (str): Name of the Optuna study.
        alpha (float): Alpha parameter for Optuna TPE sampler.

    """

    def __init__(self, config):
        super().__init__(config)
        self.n_trials = config.exp.n_trials
        self.n_startup_trials = config.exp.n_startup_trials

        self.storage = config.exp.storage
        self.study_name = config.exp.study_name
        self.alpha = config.exp.alpha

    def run(self):
        """Run the experiment with Optuna optimization.

        If the `delete_study` flag is set in the experiment configuration, the method
        deletes Optuna study data for each fold. Otherwise, it executes the experiment
        with hyperparameter tuning using Optuna.

        """

        if self.exp_config.delete_study:
            for i in range(self.n_splits):
                study_name = f"{self.study_name}_{i}"
                try:
                    optuna.delete_study(
                        study_name=study_name,
                        storage=f"sqlite:///{to_absolute_path(self.exp_config.storage)}/optuna.db",
                    )
                    print(f"Successfully deleted study {study_name}")
                except Exception:
                    print(f"study {study_name} not found.")
                if self.model_name == "rerx":
                    try:
                        study_name = f"{study_name}_pre"
                        optuna.delete_study(
                            study_name=study_name,
                            storage=f"sqlite:///{to_absolute_path(self.exp_config.storage)}/optuna.db",
                        )
                        print(f"Successfully deleted study {study_name}")
                    except Exception:
                        print(f"study {study_name} not found.")

            return
        super().run()

    def get_model_config(self, i_fold, x, y, val_data, uniq, *args, **kwargs):
        """Get the model configuration for each fold with Optuna optimization.

        This method uses the OptunaSearch class to perform hyperparameter optimization
        and retrieve the best configuration for the current fold.

        Args:
            i_fold (int): Fold index.
            x (DataFrame): Input features for training.
            y (ndarray): Target values for training.
            val_data (DataFrame): Validation data for the current fold.
            uniq (ndarray): Unique values of the target column.
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            dict: Model configuration parameters.

        """

        searcher = OptunaSearch(
            self.model_name,
            default_config=self.model_config,
            input_dim=len(self.columns),
            output_dim=len(uniq),
            X=x,
            y=y,
            val_data=val_data,
            columns=self.columns,
            target_column=self.target_column,
            onehoter=self.onehoter,
            n_trials=self.n_trials,
            n_startup_trials=self.n_startup_trials,
            storage=self.storage,
            study_name=f"{self.study_name}_{i_fold}",
            seed=self.seed,
            alpha=self.alpha,
        )
        return searcher.get_best_config()
