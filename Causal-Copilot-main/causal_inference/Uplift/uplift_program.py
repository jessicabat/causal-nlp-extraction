import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class UpliftProgram:
    """
    Class to execute uplift modeling analysis using the selected algorithm and parameters
    """
    def __init__(self, args):
        self.args = args
    
    def get_estimator(self, algo_name, params, treatment_col, outcome_col):
        """Get the appropriate estimator based on the algorithm name"""
        from causal_inference.Uplift.wrappers.causalml_wrappers import (
            CausalMLSLearner, CausalMLTLearner, CausalMLXLearner, 
            CausalMLRLearner, CausalMLUpliftTree, CausalMLUpliftRandomForest
        )
        
        estimator_map = {
            "S-Learner": CausalMLSLearner,
            "T-Learner": CausalMLTLearner,
            "X-Learner": CausalMLXLearner,
            "R-Learner": CausalMLRLearner,
            "UpliftTreeClassifier": CausalMLUpliftTree,
            "UpliftRandomForestClassifier": CausalMLUpliftRandomForest
        }
        
        estimator_class = estimator_map.get(algo_name)
        if estimator_class is None:
            print(f"Unknown algorithm: {algo_name}, defaulting to S-Learner")
            estimator_class = CausalMLSLearner
            
        return estimator_class(params, treatment_col, outcome_col)
    
    def forward(self, global_state):
        """Run the uplift modeling analysis"""
        data = global_state.user_data.processed_data
        treatment_col = global_state.inference.treatment_col
        outcome_col = global_state.inference.outcome_col
        
        # Get algorithm name and parameters
        algo_name = global_state.inference.uplift_algo_json.get('name', 'S-Learner')
        model_params = global_state.inference.uplift_model_param
        
        # Create and fit estimator
        estimator = self.get_estimator(algo_name, model_params, treatment_col, outcome_col)
        estimator.fit(data)
        
        # Predict uplift scores
        X = data.drop(columns=[treatment_col, outcome_col], errors='ignore')
        uplift_scores = estimator.predict_uplift(X)
        
        # Append uplift scores to the original data
        data['uplift_score'] = uplift_scores
        
        # Store results in global state
        global_state.inference.uplift_estimator = estimator
        global_state.inference.uplift_scores = uplift_scores
        global_state.inference.data_with_uplift = data
        
        # Generate and store visualizations
        self.generate_visualizations(global_state)
        
        return global_state
    
    def generate_visualizations(self, global_state):
        """Generate visualizations for uplift model results"""
        data = global_state.inference.data_with_uplift
        treatment_col = global_state.inference.treatment_col
        outcome_col = global_state.inference.outcome_col
        
        # Create uplift quantiles
        data['uplift_quantile'] = pd.qcut(data['uplift_score'], 5, labels=False)
        
        # Calculate average treatment effect by quantile
        ate_by_quantile = data.groupby('uplift_quantile').apply(
            lambda x: (x[x[treatment_col] == 1][outcome_col].mean() - 
                      x[x[treatment_col] == 0][outcome_col].mean())
        ).reset_index()
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x='uplift_quantile', y=0, data=ate_by_quantile)
        plt.title('Average Treatment Effect by Uplift Quantile')
        plt.xlabel('Uplift Quantile (Higher = Higher Predicted Uplift)')
        plt.ylabel('Average Treatment Effect')
        plt.tight_layout()
        
        # Save figure to global state
        global_state.inference.uplift_visualization = plt
        
        # Feature importance if available
        feature_importance = global_state.inference.uplift_estimator.feature_importance()
        if feature_importance:
            # Convert to DataFrame for visualization
            if isinstance(feature_importance, dict):
                fi_df = pd.DataFrame({
                    'Feature': list(feature_importance.keys()),
                    'Importance': list(feature_importance.values())
                })
            else:
                features = data.drop(columns=[treatment_col, outcome_col, 'uplift_score', 'uplift_quantile'], 
                                    errors='ignore').columns
                fi_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': feature_importance
                })
            
            fi_df = fi_df.sort_values('Importance', ascending=False).head(10)
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=fi_df)
            plt.title('Top 10 Feature Importance for Uplift Model')
            plt.tight_layout()
            
            # Save feature importance visualization
            global_state.inference.uplift_feature_importance = plt
        
        return global_state
    
    def qini_curve(self, global_state):
        """
        Generate a Qini curve - a common evaluation metric for uplift models
        """
        data = global_state.inference.data_with_uplift
        treatment_col = global_state.inference.treatment_col
        outcome_col = global_state.inference.outcome_col
        
        # Sort by uplift score in descending order
        sorted_data = data.sort_values('uplift_score', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative values
        n = len(sorted_data)
        n_treatment = sum(sorted_data[treatment_col] == 1)
        n_control = sum(sorted_data[treatment_col] == 0)
        
        cum_n = np.array(range(1, n + 1))
        cum_tr = np.cumsum(sorted_data[treatment_col].values)
        cum_co = cum_n - cum_tr
        
        cum_tr_outcomes = np.cumsum(sorted_data[treatment_col] * sorted_data[outcome_col])
        cum_co_outcomes = np.cumsum((1 - sorted_data[treatment_col]) * sorted_data[outcome_col])
        
        # Calculate Qini values
        qini_y = cum_tr_outcomes / np.maximum(cum_tr, 1) - cum_co_outcomes / np.maximum(cum_co, 1)
        qini_x = cum_n / n
        
        # Calculate random selection baseline
        random_y = np.linspace(0, 
                             (sorted_data[sorted_data[treatment_col] == 1][outcome_col].mean() - 
                              sorted_data[sorted_data[treatment_col] == 0][outcome_col].mean()), 
                             len(qini_x))
        
        # Create Qini curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(qini_x, qini_y, label='Uplift Model')
        plt.plot(qini_x, random_y, label='Random Selection', linestyle='--')
        plt.fill_between(qini_x, random_y, qini_y, alpha=0.2, where=(qini_y >= random_y))
        plt.title('Qini Curve')
        plt.xlabel('Fraction of population targeted')
        plt.ylabel('Incremental outcome')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Calculate Qini coefficient (area between model curve and random baseline)
        qini_coef = np.trapz(qini_y, qini_x) - np.trapz(random_y, qini_x)
        
        # Save to global state
        global_state.inference.qini_visualization = plt
        global_state.inference.qini_coefficient = qini_coef
        
        return global_state