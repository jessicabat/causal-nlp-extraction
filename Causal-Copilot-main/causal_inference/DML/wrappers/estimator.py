import numpy as np
import pandas as pd
from typing import Dict, Tuple

from econml.dml import DML as Econ_DML
from econml.dml import LinearDML as Econ_LinearDML
from econml.dml import SparseLinearDML as Econ_SparseLinearDML
from econml.dml import CausalForestDML as Econ_CausalForestDML

from .base import Estimator

# - DML 
# - LinearDML
# - SparseLinearDML
# - CausalForestDML

class DML(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int=0, T1: int=1):
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        self.model = Econ_DML(**self._params)

    @property
    def name(self):
        return "DML"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col]
        W = data[self.W_col]
        # Run DML algorithm        
        self.model.fit(y, T, X=X, W=W)
    
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        # ✅ Use isclose for float-safe comparison
        treated_indices =  np.isclose(data[self.T_col], self.T1)
        # treated_indices = (data[self.T_col] == self.T1)

        # ✅ Handle case when no treated units are found
        if treated_indices.sum() == 0:
            print(f"[WARN] No treated samples found for T1 = {self.T1}. ATT cannot be computed.")
            return np.nan, np.nan, np.nan
        
        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass

class LinearDML(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int=0, T1: int=1):
        params.pop('model_final', None)
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        self.model = Econ_LinearDML(**self._params)

    @property
    def name(self):
        return "LinearDML"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col]
        W = data[self.W_col]
        # Run DML algorithm        
        self.model.fit(y, T, X=X, W=W)
        
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        # ✅ Use isclose for float-safe comparison
        treated_indices =  np.isclose(data[self.T_col], self.T1)
        # treated_indices = (data[self.T_col] == self.T1)

        # ✅ Handle case when no treated units are found
        if treated_indices.sum() == 0:
            print(f"[WARN] No treated samples found for T1 = {self.T1}. ATT cannot be computed.")
            return np.nan, np.nan, np.nan
        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass

class SparseLinearDML(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int=0, T1: int=1):
        params.pop('model_final', None)
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        self.model = Econ_SparseLinearDML(**self._params)

    @property
    def name(self):
        return "SparseLinearDML"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col]
        W = data[self.W_col]
        # Run DML algorithm        
        self.model.fit(y, T, X=X, W=W)
           
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        # ✅ Use isclose for float-safe comparison
        treated_indices =  np.isclose(data[self.T_col], self.T1)
        # treated_indices = (data[self.T_col] == self.T1)

        # ✅ Handle case when no treated units are found
        if treated_indices.sum() == 0:
            print(f"[WARN] No treated samples found for T1 = {self.T1}. ATT cannot be computed.")
            return np.nan, np.nan, np.nan
        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass

class CausalForestDML(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int = 0, T1: int = 1):
        params.pop('model_final', None)
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        self.model = Econ_CausalForestDML(**self._params)

    @property
    def name(self):
        return "CausalForestDML"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col]
        W = data[self.W_col]
        # Run DML algorithm        
        self.model.fit(y, T, X=X, W=W)
    
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        # ✅ Use isclose for float-safe comparison
        treated_indices =  np.isclose(data[self.T_col], self.T1)
        # treated_indices = (data[self.T_col] == self.T1)

        # ✅ Handle case when no treated units are found
        if treated_indices.sum() == 0:
            print(f"[WARN] No treated samples found for T1 = {self.T1}. ATT cannot be computed.")
            return np.nan, np.nan, np.nan
        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass


if __name__ == "__main__":
    pc_algo = DML(y_col='', T_col='', X_col='')
    pc_algo.test_algorithm() 