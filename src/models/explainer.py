# explainer.py - Model Explainability with SHAP and LIME
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class TransactionExplainer:
    """
    A comprehensive explainer class for transaction risk models using SHAP and LIME.
    Supports preprocessing, feature extraction, and visualization.
    """
    
    def __init__(self, model_path: str, scaler_path: str, feature_names: Optional[List[str]] = None):
        """
        Initialize the explainer with model and scaler paths.
        
        Args:
            model_path (str): Path to the trained model pickle file
            scaler_path (str): Path to the scaler pickle file
            feature_names (List[str], optional): List of feature names used in training
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = feature_names or self._get_default_features()
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Load model and scaler
        self._load_model_and_scaler()
        
    def _get_default_features(self) -> List[str]:
        """Default feature names for AML transaction data."""
        return [
            "Amount_Received", "Amount_Paid", "Currency_Mismatch", 
            "Receiving_Currency", "Payment_Currency", "Payment_Format",
            "Transaction_Hour", "Transaction_Day", "Account_Age_Days",
            "Previous_Transaction_Count", "Average_Transaction_Amount",
            "Velocity_1h", "Velocity_24h", "High_Risk_Country"
        ]
    
    def _load_model_and_scaler(self):
        """Load the trained model and scaler."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✓ Model loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"✓ Scaler loaded from {self.scaler_path}")
            else:
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
                
        except Exception as e:
            print(f"❌ Error loading model/scaler: {e}")
            raise
    
    def _initialize_explainers(self, training_data: Optional[np.ndarray] = None):
        """Initialize SHAP and LIME explainers."""
        try:
            # Initialize SHAP explainer
            if hasattr(self.model, 'predict_proba'):
                self.shap_explainer = shap.Explainer(self.model)
            else:
                # For models without predict_proba, use TreeExplainer or KernelExplainer
                self.shap_explainer = shap.TreeExplainer(self.model)
            
            # Initialize LIME explainer
            if training_data is None:
                # Create dummy training data if none provided
                training_data = np.random.normal(0, 1, (100, len(self.feature_names)))
            
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=self.feature_names,
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )
            
            print("✓ SHAP and LIME explainers initialized")
            
        except Exception as e:
            print(f"❌ Error initializing explainers: {e}")
            raise
    
    def preprocess_transaction(self, transaction: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess transaction data for model input.
        
        Args:
            transaction (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Preprocessed transaction data
        """
        try:
            # Make a copy to avoid modifying original data
            processed_df = transaction.copy()
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(processed_df.columns)
            if missing_features:
                print(f"⚠️ Missing features: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    processed_df[feature] = 0
            
            # Select only the required features in correct order
            processed_df = processed_df[self.feature_names]
            
            # Handle missing values
            processed_df = processed_df.fillna(0)
            
            # Ensure correct data types
            for col in processed_df.columns:
                if processed_df[col].dtype == 'object':
                    try:
                        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                        processed_df[col] = processed_df[col].fillna(0)
                    except:
                        # If conversion fails, use label encoding
                        processed_df[col] = pd.Categorical(processed_df[col]).codes
            
            return processed_df
            
        except Exception as e:
            print(f"❌ Error preprocessing transaction: {e}")
            raise
    
    def explain_with_shap(self, transaction: pd.DataFrame, top_k: int = 5) -> Dict[str, float]:
        """
        Explain a transaction using SHAP.
        
        Args:
            transaction (pd.DataFrame): Preprocessed transaction data
            top_k (int): Number of top contributing features to return
            
        Returns:
            Dict[str, float]: Dictionary of feature names and their SHAP values
        """
        try:
            if self.shap_explainer is None:
                self._initialize_explainers()
            
            # Scale the features
            X_scaled = self.scaler.transform(transaction)
            
            # Get SHAP values
            shap_values = self.shap_explainer(X_scaled)
            
            # Extract values for the first (and likely only) instance
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) > 2:
                    # For multi-class classification, take the positive class
                    shap_vals = shap_values.values[0][:, 1]
                else:
                    shap_vals = shap_values.values[0]
            else:
                shap_vals = shap_values[0]
            
            # Get top contributing features
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(shap_vals):
                    feature_importance[feature] = float(shap_vals[i])
            
            # Sort by absolute importance and return top_k
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:top_k]
            
            return dict(sorted_features)
            
        except Exception as e:
            print(f"❌ Error in SHAP explanation: {e}")
            raise
    
    def explain_with_lime(self, transaction: pd.DataFrame, top_k: int = 5) -> Dict[str, float]:
        """
        Explain a transaction using LIME.
        
        Args:
            transaction (pd.DataFrame): Preprocessed transaction data
            top_k (int): Number of top contributing features to return
            
        Returns:
            Dict[str, float]: Dictionary of feature names and their LIME coefficients
        """
        try:
            if self.lime_explainer is None:
                # Initialize with scaled training data
                dummy_data = np.random.normal(0, 1, (100, len(self.feature_names)))
                scaled_dummy = self.scaler.transform(dummy_data)
                self._initialize_explainers(scaled_dummy)
            
            # Scale the features
            X_scaled = self.scaler.transform(transaction)
            
            # Get LIME explanation
            explanation = self.lime_explainer.explain_instance(
                X_scaled[0],
                self.model.predict_proba,
                num_features=top_k,
                num_samples=1000
            )
            
            # Extract feature contributions
            feature_contributions = dict(explanation.as_list())
            
            return feature_contributions
            
        except Exception as e:
            print(f"❌ Error in LIME explanation: {e}")
            raise
    
    def get_explanation(self, transaction: pd.DataFrame, method: str = "shap", 
                      top_k: int = 5, preprocess: bool = True) -> Dict[str, float]:
        """
        Get explanation for a transaction using specified method.
        
        Args:
            transaction (pd.DataFrame): Transaction data
            method (str): Explanation method ('shap' or 'lime')
            top_k (int): Number of top features to return
            preprocess (bool): Whether to preprocess the transaction
            
        Returns:
            Dict[str, float]: Feature contributions
        """
        try:
            # Preprocess if needed
            if preprocess:
                processed_transaction = self.preprocess_transaction(transaction)
            else:
                processed_transaction = transaction
            
            # Get prediction for context
            X_scaled = self.scaler.transform(processed_transaction)
            prediction = self.model.predict(X_scaled)[0]
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            
            print(f"🔍 Transaction Prediction: {prediction}")
            print(f"📊 Prediction Probability: {prediction_proba}")
            
            # Get explanation based on method
            if method.lower() == "shap":
                explanation = self.explain_with_shap(processed_transaction, top_k)
            elif method.lower() == "lime":
                explanation = self.explain_with_lime(processed_transaction, top_k)
            else:
                raise ValueError("Method must be 'shap' or 'lime'")
            
            return explanation
            
        except Exception as e:
            print(f"❌ Error getting explanation: {e}")
            raise
    
    def plot_explanation(self, transaction: pd.DataFrame, method: str = "shap", 
                        top_k: int = 5, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate visualization of feature contributions.
        
        Args:
            transaction (pd.DataFrame): Transaction data
            method (str): Explanation method ('shap' or 'lime')
            top_k (int): Number of top features to show
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The generated plot figure
        """
        try:
            # Get explanation
            explanation = self.get_explanation(transaction, method, top_k)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            features = list(explanation.keys())
            values = list(explanation.values())
            
            # Create horizontal bar plot
            colors = ['red' if v < 0 else 'green' for v in values]
            bars = ax.barh(features, values, color=colors, alpha=0.7)
            
            # Customize plot
            ax.set_xlabel(f'{method.upper()} Feature Contribution')
            ax.set_title(f'Top {top_k} Feature Contributions ({method.upper()})')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax.text(width + (0.01 if width >= 0 else -0.01), 
                       bar.get_y() + bar.get_height()/2, 
                       f'{value:.4f}', 
                       ha='left' if width >= 0 else 'right',
                       va='center')
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating plot: {e}")
            raise
    
    def compare_explanations(self, transaction: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """
        Compare SHAP and LIME explanations side by side.
        
        Args:
            transaction (pd.DataFrame): Transaction data
            top_k (int): Number of top features to compare
            
        Returns:
            pd.DataFrame: Comparison of SHAP and LIME explanations
        """
        try:
            shap_explanation = self.get_explanation(transaction, "shap", top_k)
            lime_explanation = self.get_explanation(transaction, "lime", top_k)
            
            # Get all unique features
            all_features = set(list(shap_explanation.keys()) + list(lime_explanation.keys()))
            
            # Create comparison dataframe
            comparison_data = []
            for feature in all_features:
                comparison_data.append({
                    'Feature': feature,
                    'SHAP_Value': shap_explanation.get(feature, 0),
                    'LIME_Value': lime_explanation.get(feature, 0)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('SHAP_Value', key=abs, ascending=False)
            
            return comparison_df
            
        except Exception as e:
            print(f"❌ Error comparing explanations: {e}")
            raise


def main():
    """
    Example usage of the TransactionExplainer class.
    """
    # Example configuration
    MODEL_PATH = "models/risk_classifier_xgb.pkl"
    SCALER_PATH = "models/risk_classifier_scaler.pkl"
    
    # Create sample transaction data
    sample_transaction = pd.DataFrame({
        'Amount_Received': [5000.0],
        'Amount_Paid': [5000.0],
        'Currency_Mismatch': [0],
        'Receiving_Currency': [1],
        'Payment_Currency': [1],
        'Payment_Format': [2],
        'Transaction_Hour': [14],
        'Transaction_Day': [3],
        'Account_Age_Days': [365],
        'Previous_Transaction_Count': [10],
        'Average_Transaction_Amount': [2500.0],
        'Velocity_1h': [1],
        'Velocity_24h': [3],
        'High_Risk_Country': [0]
    })
    
    try:
        # Initialize explainer
        explainer = TransactionExplainer(MODEL_PATH, SCALER_PATH)
        
        # Get SHAP explanation
        print("\n=== SHAP Explanation ===")
        shap_result = explainer.get_explanation(sample_transaction, method="shap")
        for feature, value in shap_result.items():
            print(f"{feature}: {value:.4f}")
        
        # Get LIME explanation
        print("\n=== LIME Explanation ===")
        lime_result = explainer.get_explanation(sample_transaction, method="lime")
        for feature, value in lime_result.items():
            print(f"{feature}: {value:.4f}")
        
        # Compare explanations
        print("\n=== Comparison ===")
        comparison = explainer.compare_explanations(sample_transaction)
        print(comparison)
        
        # Generate visualizations
        print("\n=== Generating Plots ===")
        shap_fig = explainer.plot_explanation(sample_transaction, method="shap")
        lime_fig = explainer.plot_explanation(sample_transaction, method="lime")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")


if __name__ == "__main__":
    main()