from typing import *
import pandas as pd
import numpy as np
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from .base import UpliftEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier # Add GB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier # Add single trees
from causal_inference.Uplift.wrappers.base import UpliftEstimator

class CausalMLSLearner(UpliftEstimator):
    def __init__(self, params: Dict, treatment_col: str, outcome_col: str):
        super().__init__(params, treatment_col, outcome_col)
        # Use a default *tree-based* model if none is provided
        learner = params.get('model', RandomForestRegressor())
        self.model = BaseSRegressor(learner=learner)
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col

    def fit(self, data: pd.DataFrame):
        X = data.drop(columns=[self.treatment_col, self.outcome_col], errors='ignore')
        T = data[self.treatment_col]
        y = data[self.outcome_col]
        self.model.fit(X=X.values, y=y.values, t=T.values)

    def predict_uplift(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

class CausalMLTLearner(UpliftEstimator):
    def __init__(self, params: Dict, treatment_col: str, outcome_col: str):
        super().__init__(params, treatment_col, outcome_col)
        # Default to RandomForestRegressor
        learner = params.get('model', RandomForestRegressor())
        self.model = BaseTRegressor(learner=learner)
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col

    def fit(self, data: pd.DataFrame):
        X = data.drop(columns=[self.treatment_col, self.outcome_col], errors='ignore')
        T = data[self.treatment_col]
        y = data[self.outcome_col]
        self.model.fit(X=X.values, y=y.values, t=T.values)

    def predict_uplift(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

class CausalMLXLearner(UpliftEstimator):
    def __init__(self, params: Dict, treatment_col: str, outcome_col: str):
        super().__init__(params, treatment_col, outcome_col)
        # Default to RandomForestRegressor
        learner = params.get('model', RandomForestRegressor())
        self.model = BaseXRegressor(learner=learner)
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col

    def fit(self, data: pd.DataFrame):
        X = data.drop(columns=[self.treatment_col, self.outcome_col], errors='ignore')
        T = data[self.treatment_col]
        y = data[self.outcome_col]
        self.model.fit(X=X.values, y=y.values, t=T.values)

    def predict_uplift(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

class CausalMLRLearner(UpliftEstimator):
    def __init__(self, params: Dict, treatment_col: str, outcome_col: str):
        super().__init__(params, treatment_col, outcome_col)
        # Default to RandomForestRegressor
        learner = params.get('model', RandomForestRegressor())
        self.model = BaseRRegressor(learner=learner)
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col

    def fit(self, data: pd.DataFrame):
        X = data.drop(columns=[self.treatment_col, self.outcome_col], errors='ignore')
        T = data[self.treatment_col]
        y = data[self.outcome_col]
        self.model.fit(X=X.values, y=y.values, t=T.values)

    def predict_uplift(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)


class CausalMLUpliftTree(UpliftEstimator):
    def __init__(self, params: Dict, treatment_col:str, outcome_col:str):
        super().__init__(params, treatment_col, outcome_col)
        # UpliftTreeClassifier expects specific parameters.
        # Use get() with defaults for robustness.
        self.model = UpliftTreeClassifier(
            control_name='0',  # MUST be a string
            max_depth=params.get('max_depth', None),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            min_samples_treatment=params.get('min_samples_treatment', 1),
            n_reg=params.get('n_reg', 100),
            evaluationFunction=params.get('evaluationFunction', 'KL'),
            splitter=params.get('splitter', 'best'),
            # Add other parameters as needed, with defaults.
        )
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col

    def fit(self, data: pd.DataFrame):
        X = data.drop(columns=[self.treatment_col, self.outcome_col], errors='ignore')
        T = data[self.treatment_col]
        y = data[self.outcome_col]
        # VERY IMPORTANT: Convert treatment to string for UpliftTreeClassifier
        T = T.astype(str)
        self.model.fit(X=X.values, y=y.values, t=T.values)


    def predict_uplift(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

class CausalMLUpliftRandomForest(UpliftEstimator):
    def __init__(self, params: Dict, treatment_col: str, outcome_col: str):
        super().__init__(params, treatment_col, outcome_col)
        self.model = UpliftRandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            control_name='0',  # MUST be a string
            max_depth=params.get('max_depth', None),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            min_samples_treatment=params.get('min_samples_treatment', 1),
            n_reg=params.get('n_reg', 100),
            evaluationFunction=params.get('evaluationFunction', 'KL'),
            # Add other parameters with defaults
        )
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col

    def fit(self, data: pd.DataFrame):
        X = data.drop(columns=[self.treatment_col, self.outcome_col], errors='ignore')
        T = data[self.treatment_col]
        y = data[self.outcome_col]
        # VERY IMPORTANT: Convert treatment to string
        T = T.astype(str)
        self.model.fit(X=X.values, y=y.values, t=T.values)

    def predict_uplift(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)