import numpy as np
import pandas as pd
from typing import Dict, Tuple

from econml.iv.dr import DRIV as Econ_DRIV
from econml.iv.dr import LinearDRIV as Econ_LinearDRIV
from econml.iv.dr import SparseLinearDRIV as Econ_SparseLinearDRIV
from econml.iv.dr import ForestDRIV as Econ_ForestDRIV

from .base import Estimator

# - DRIV 
# - LinearDRIV 
# - SparseLinearDRIV 
# - ForestDRIV 

class DRIV(Estimator):
    def __init__(self, y_col: str, T_col: str, Z_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int = 0, T1: int = 1):
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        self.Z_col = Z_col  
        self.model = Econ_DRIV(**self._params)

    @property
    def name(self):
        return "DRIV"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col]
        Z = data[[self.Z_col]]
        W = data[self.W_col]
        # Run DRIV algorithm        
        self.model.fit(y, T, X=X, Z=Z, W=W)
    
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        treated_indices = (data[self.T_col] == 1)
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

class LinearDRIV(Estimator):
    def __init__(self, y_col: str, T_col: str, Z_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int = 0, T1: int = 1):
        # Remove final stage key if present
        if 'model_final' in params:
            del params['model_final']
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        self.Z_col = Z_col
        self.model = Econ_LinearDRIV(**self._params)

    @property
    def name(self):
        return "LinearDRIV"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col]
        Z = data[[self.Z_col]]
        W = data[self.W_col] if self.W_col is not None else None
        self.model.fit(y, T, X=X, Z=Z, W=W)

    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        treated_indices = (data[self.T_col] == 1)
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

class SparseLinearDRIV(Estimator):
    def __init__(self, y_col: str, T_col: str, Z_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int = 0, T1: int = 1):
        if 'model_final' in params:
            del params['model_final']
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        self.Z_col = Z_col
        self.model = Econ_SparseLinearDRIV(**self._params)

    @property
    def name(self):
        return "SparseLinearDRIV"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col]
        Z = data[[self.Z_col]]
        W = data[self.W_col] if self.W_col is not None else None
        self.model.fit(y, T, X=X, Z=Z, W=W)

    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        treated_indices = (data[self.T_col] == 1)
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

class ForestDRIV(Estimator):
    def __init__(self, y_col: str, T_col: str, Z_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int = 0, T1: int = 1):
        if 'model_final' in params:
            del params['model_final']
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        self.Z_col = Z_col
        self.model = Econ_ForestDRIV(**self._params)

    @property
    def name(self):
        return "ForestDRIV"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col]
        Z = data[[self.Z_col]]
        W = data[self.W_col] if self.W_col is not None else None
        self.model.fit(y, T, X=X, Z=Z, W=W)

    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        treated_indices = (data[self.T_col] == 1)
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
    pc_algo = DRIV(y_col='', T_col='', Z_col='', X_col='')
    pc_algo.test_algorithm() 