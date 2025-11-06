from typing import Dict, Union, Any
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class UpliftEstimator(ABC):
    """Base class for all Uplift Modeling estimators"""
    
    def __init__(self, params: Dict, treatment_col: str, outcome_col: str):
        self.params = params
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Fit the model on the provided data"""
        pass
        
    @abstractmethod
    def predict_uplift(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the uplift/CATE for new samples"""
        pass
        
    def feature_importance(self) -> Dict[str, float]:
        """Return feature importance if the model supports it"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'learner') and hasattr(self.model.learner, 'feature_importances_'):
            return self.model.learner.feature_importances_
        else:
            return {}