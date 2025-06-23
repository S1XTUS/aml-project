import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.load_data import load_and_clean_data
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed paths - use relative paths and os.path.join for cross-platform compatibility
# Get the project root directory (go up two levels from src/models/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "risk_classifier_xgb.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "aml_preprocessor.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "risk_classifier_scaler.pkl")  # Explicit scaler path
FEATURE_STATS_PATH = os.path.join(MODELS_DIR, "aml_feature_stats.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "aml_encoders.pkl")

class AdvancedAMLRiskClassifier:
    """
    Advanced AML Risk Classifier for transaction monitoring
    Handles the complete pipeline from training to prediction
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.preprocessor_path = PREPROCESSOR_PATH
        self.scaler_path = SCALER_PATH  # Explicit scaler path
        self.feature_stats_path = FEATURE_STATS_PATH
        self.encoders_path = ENCODERS_PATH
        
        # Model components
        self.model = None
        self.preprocessor = None
        self.scaler = None  # Explicit scaler reference
        self.feature_stats = {}
        self.encoders = {}
        self.feature_columns = []
        self.optimal_threshold = 0.5
        
        # High-risk patterns
        self.high_risk_currencies = {'USD', 'EUR', 'CHF', 'GBP'}  # Common money laundering currencies
        self.suspicious_amounts = [10000, 9000, 8000, 5000]  # Common structuring amounts
        
        # Initialize with basic defaults if no model exists
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize basic components for when no trained model exists"""
        if not os.path.exists(self.model_path):
            logger.info("No trained model found. Initializing with defaults.")
            # Basic feature columns that would be created
            self.feature_columns = [
                'hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'is_night_transaction',
                'month', 'quarter', 'amount_ratio', 'amount_difference', 'amount_difference_pct',
                'currency_match', 'involves_high_risk_currency', 'potential_structuring',
                'is_round_amount', 'same_bank', 'Amount_Paid', 'Amount_Received',
                'From_Bank', 'To_Bank'
            ]
            
            # Initialize basic encoders
            self.encoders = {
                'Receiving_Currency': LabelEncoder(),
                'Payment_Currency': LabelEncoder(),
                'Payment_Format': LabelEncoder(),
                'bank_pair': LabelEncoder()
            }
            
            # Fit encoders with common values
            self.encoders['Receiving_Currency'].fit(['USD', 'EUR', 'GBP', 'CHF', 'Unknown'])
            self.encoders['Payment_Currency'].fit(['USD', 'EUR', 'GBP', 'CHF', 'Unknown'])
            self.encoders['Payment_Format'].fit(['wire_transfer', 'card', 'check', 'cash', 'Unknown'])
            self.encoders['bank_pair'].fit(['0_0', '1_1', '123_456', 'Unknown'])
            
            # Initialize scaler
            self.scaler = StandardScaler()
            self.preprocessor = self.scaler  # Keep backward compatibility
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features from the transaction data
        """
        logger.info("Creating advanced features...")
        df_featured = df.copy()
        
        # Ensure Timestamp is datetime
        if 'Timestamp' in df_featured.columns:
            df_featured['Timestamp'] = pd.to_datetime(df_featured['Timestamp'])
        else:
            df_featured['Timestamp'] = pd.to_datetime('now')
        
        # Time-based features
        df_featured['hour'] = df_featured['Timestamp'].dt.hour
        df_featured['day_of_week'] = df_featured['Timestamp'].dt.dayofweek
        df_featured['is_weekend'] = (df_featured['day_of_week'] >= 5).astype(int)
        df_featured['is_business_hours'] = ((df_featured['hour'] >= 9) & (df_featured['hour'] <= 17)).astype(int)
        df_featured['is_night_transaction'] = ((df_featured['hour'] <= 6) | (df_featured['hour'] >= 22)).astype(int)
        df_featured['month'] = df_featured['Timestamp'].dt.month
        df_featured['quarter'] = df_featured['Timestamp'].dt.quarter
        
        # Amount-based features
        df_featured['amount_ratio'] = df_featured['Amount_Received'] / (df_featured['Amount_Paid'] + 1e-6)
        df_featured['amount_difference'] = abs(df_featured['Amount_Received'] - df_featured['Amount_Paid'])
        df_featured['amount_difference_pct'] = df_featured['amount_difference'] / (df_featured['Amount_Paid'] + 1e-6)
        
        # Currency features
        df_featured['currency_match'] = (df_featured['Receiving_Currency'] == df_featured['Payment_Currency']).astype(int)
        df_featured['involves_high_risk_currency'] = df_featured.apply(
            lambda x: int(x['Receiving_Currency'] in self.high_risk_currencies or 
                         x['Payment_Currency'] in self.high_risk_currencies), axis=1
        )
        
        # Structuring detection (amounts close to reporting thresholds)
        df_featured['potential_structuring'] = df_featured['Amount_Paid'].apply(
            lambda x: int(any(abs(x - amt) <= 500 for amt in self.suspicious_amounts))
        )
        
        # Round number detection
        df_featured['is_round_amount'] = ((df_featured['Amount_Paid'] % 1000) == 0).astype(int)
        
        # Bank relationship features
        df_featured['same_bank'] = (df_featured['From_Bank'] == df_featured['To_Bank']).astype(int)
        df_featured['bank_pair'] = df_featured['From_Bank'].astype(str) + '_' + df_featured['To_Bank'].astype(str)
        
        # Only create complex features if we have sufficient data
        if len(df_featured) > 1:
            # Account activity features (requires sorting by timestamp first)
            df_featured = df_featured.sort_values(['From_Account', 'Timestamp'])
            
            # Velocity features - transactions per account in different time windows
            for window in ['1D', '7D', '30D']:
                try:
                    df_featured[f'from_account_tx_count_{window}'] = df_featured.groupby('From_Account')['Amount_Paid'].rolling(
                        window=window, on='Timestamp').count().reset_index(0, drop=True)
                    
                    df_featured[f'from_account_tx_sum_{window}'] = df_featured.groupby('From_Account')['Amount_Paid'].rolling(
                        window=window, on='Timestamp').sum().reset_index(0, drop=True)
                    
                    df_featured[f'to_account_tx_count_{window}'] = df_featured.groupby('To_Account')['Amount_Received'].rolling(
                        window=window, on='Timestamp').count().reset_index(0, drop=True)
                except:
                    # If rolling window fails, use simple counts
                    df_featured[f'from_account_tx_count_{window}'] = 1
                    df_featured[f'from_account_tx_sum_{window}'] = df_featured['Amount_Paid']
                    df_featured[f'to_account_tx_count_{window}'] = 1
            
            # Historical patterns for each account
            try:
                account_stats = df_featured.groupby('From_Account').agg({
                    'Amount_Paid': ['mean', 'std', 'count'],
                    'hour': 'mean',
                    'same_bank': 'mean'
                }).reset_index()
                
                # Flatten column names
                account_stats.columns = ['From_Account'] + [f'from_account_{col[0]}_{col[1]}' if col[1] else f'from_account_{col[0]}' 
                                                           for col in account_stats.columns[1:]]
                
                df_featured = df_featured.merge(account_stats, on='From_Account', how='left')
                
                # Deviation from historical patterns
                df_featured['amount_zscore'] = (df_featured['Amount_Paid'] - df_featured['from_account_Amount_Paid_mean']) / (df_featured['from_account_Amount_Paid_std'] + 1e-6)
                df_featured['hour_deviation'] = abs(df_featured['hour'] - df_featured['from_account_hour_mean'])
            except:
                # Default values if aggregation fails
                df_featured['amount_zscore'] = 0
                df_featured['hour_deviation'] = 0
            
            # Network features
            try:
                unique_recipients = df_featured.groupby('From_Account')['To_Account'].nunique().reset_index()
                unique_recipients.columns = ['From_Account', 'unique_recipients_count']
                df_featured = df_featured.merge(unique_recipients, on='From_Account', how='left')
                
                unique_senders = df_featured.groupby('To_Account')['From_Account'].nunique().reset_index()
                unique_senders.columns = ['To_Account', 'unique_senders_count']
                df_featured = df_featured.merge(unique_senders, on='To_Account', how='left')
            except:
                df_featured['unique_recipients_count'] = 1
                df_featured['unique_senders_count'] = 1
        else:
            # Default values for single transaction
            for window in ['1D', '7D', '30D']:
                df_featured[f'from_account_tx_count_{window}'] = 1
                df_featured[f'from_account_tx_sum_{window}'] = df_featured['Amount_Paid']
                df_featured[f'to_account_tx_count_{window}'] = 1
            
            df_featured['amount_zscore'] = 0
            df_featured['hour_deviation'] = 0
            df_featured['unique_recipients_count'] = 1
            df_featured['unique_senders_count'] = 1
        
        # Payment format risk
        if len(df_featured) > 10:  # Only if we have enough data
            payment_format_counts = df_featured['Payment_Format'].value_counts()
            rare_formats = payment_format_counts[payment_format_counts < df_featured.shape[0] * 0.01].index
            df_featured['rare_payment_format'] = df_featured['Payment_Format'].isin(rare_formats).astype(int)
        else:
            df_featured['rare_payment_format'] = 0
        
        logger.info(f"Created {len(df_featured.columns) - len(df.columns)} new features")
        return df_featured
    
    def prepare_features_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for model training
        """
        logger.info("Preparing features for training...")
        
        # Create advanced features
        df_featured = self.create_advanced_features(df)
        
        # Handle categorical variables
        categorical_columns = ['Receiving_Currency', 'Payment_Currency', 'Payment_Format', 'bank_pair']
        
        for col in categorical_columns:
            if col in df_featured.columns:
                # Create encoder
                encoder = LabelEncoder()
                df_featured[f'{col}_encoded'] = encoder.fit_transform(df_featured[col].fillna('Unknown'))
                self.encoders[col] = encoder
        
        # Select numeric features for modeling
        exclude_columns = ['Timestamp', 'From_Account', 'To_Account', 'Receiving_Currency', 
                          'Payment_Currency', 'Payment_Format', 'bank_pair', 'Is_Laundering']
        
        feature_columns = [col for col in df_featured.columns if col not in exclude_columns]
        
        # Handle missing values
        df_featured[feature_columns] = df_featured[feature_columns].fillna(0)
        
        # Store feature statistics
        self.feature_stats = {
            col: {
                'mean': df_featured[col].mean(),
                'std': df_featured[col].std(),
                'min': df_featured[col].min(),
                'max': df_featured[col].max(),
                'median': df_featured[col].median()
            } for col in feature_columns
        }
        
        self.feature_columns = feature_columns
        
        logger.info(f"Prepared {len(feature_columns)} features for training")
        return df_featured[feature_columns], feature_columns
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the AML risk classification model
        """
        logger.info(f"Starting model training on {len(df)} transactions...")
        logger.info(f"Laundering cases: {df['Is_Laundering'].sum()} ({df['Is_Laundering'].mean()*100:.3f}%)")
        
        # Prepare features
        X, feature_columns = self.prepare_features_for_training(df)
        y = df['Is_Laundering']
        
        # Split data stratified
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Create and fit scaler first
        self.scaler = StandardScaler()
        self.preprocessor = self.scaler  # Keep backward compatibility
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Scaler fitted and data scaled successfully")
        
        # Handle severe class imbalance with adaptive sampling strategy
        positive_count = y_train.sum()
        negative_count = len(y_train) - positive_count
        
        logger.info(f"Original class distribution - Positive: {positive_count}, Negative: {negative_count}")
        
        # Use a more conservative approach for severe imbalance
        # First, use SMOTE to increase minority class
        k_neighbors = min(5, positive_count - 1) if positive_count > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        positive_after_smote = y_train_resampled.sum()
        negative_after_smote = len(y_train_resampled) - positive_after_smote
        
        logger.info(f"After SMOTE - Positive: {positive_after_smote}, Negative: {negative_after_smote}")
        
        # Only apply undersampling if we have enough samples and the ratio makes sense
        if negative_after_smote > positive_after_smote * 5:  # Only if majority is 5x larger
            # Calculate a reasonable ratio - aim for 1:3 or 1:4 ratio (positive:negative)
            target_ratio = min(0.25, positive_after_smote / negative_after_smote * 2)  # At most 25% positive
            
            logger.info(f"Applying undersampling with target ratio: {target_ratio:.3f}")
            
            undersampler = RandomUnderSampler(random_state=42, sampling_strategy=target_ratio)
            X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_resampled, y_train_resampled)
        else:
            logger.info("Skipping undersampling - ratio already reasonable after SMOTE")
        
        final_positive = y_train_resampled.sum()
        final_negative = len(y_train_resampled) - final_positive
        
        logger.info(f"Final resampled distribution - Positive: {final_positive}, Negative: {final_negative}")
        logger.info(f"Final positive class ratio: {y_train_resampled.mean()*100:.1f}%")
        
        # For extremely imbalanced data, use class weights as well
        pos_weight = negative_count / positive_count if positive_count > 0 else 1
        scale_pos_weight = min(pos_weight, 100)  # Cap the weight to avoid extreme values
        
        # Train XGBoost model with appropriate parameters for imbalanced data
        self.model = xgb.XGBClassifier(
            n_estimators=200,  # Reduced for faster training with large dataset
            max_depth=6,       # Reduced to prevent overfitting
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            early_stopping_rounds=20,  # Reduced for faster training
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight  # Handle remaining imbalance
        )
        
        logger.info(f"Training XGBoost with scale_pos_weight={scale_pos_weight:.2f}")
        
        # Fit model with early stopping
        self.model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate model
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Find optimal threshold using precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # For imbalanced data, optimize for F1 score or F2 score (emphasizes recall)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        f2_scores = 5 * (precision * recall) / (4 * precision + recall + 1e-8)  # Emphasizes recall more
        
        # Use F2 score for AML (better to catch more suspicious transactions)
        optimal_threshold = thresholds[np.argmax(f2_scores)]
        
        # Store optimal threshold
        self.optimal_threshold = optimal_threshold
        
        # Additional evaluation metrics
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision_optimal = precision_score(y_test, y_pred_optimal)
        recall_optimal = recall_score(y_test, y_pred_optimal)
        f1_optimal = f1_score(y_test, y_pred_optimal)
        
        # Feature importance
        feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        
        results = {
            'auc_score': auc_score,
            'optimal_threshold': optimal_threshold,
            'precision': precision_optimal,
            'recall': recall_optimal,
            'f1_score': f1_optimal,
            'top_features': top_features,
            'training_samples': len(X_train_resampled),
            'test_samples': len(X_test),
            'feature_count': len(feature_columns),
            'original_positive_ratio': positive_count / len(y_train),
            'final_positive_ratio': y_train_resampled.mean(),
            'scale_pos_weight': scale_pos_weight
        }
        
        logger.info(f"Model training completed!")
        logger.info(f"AUC Score: {auc_score:.4f}")
        logger.info(f"Optimal Threshold: {optimal_threshold:.4f}")
        logger.info(f"Precision: {precision_optimal:.4f}")
        logger.info(f"Recall: {recall_optimal:.4f}")
        logger.info(f"F1 Score: {f1_optimal:.4f}")
        logger.info(f"Top 5 features: {[f[0] for f in top_features[:5]]}")
        
        return results

    
    def save_model(self, model_path: str = None):
        """
        Save the trained model and all components
        """
        if model_path is None:
            model_path = self.model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler explicitly to the specified path
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Scaler saved to {self.scaler_path}")
        
        # Save preprocessor (for backward compatibility)
        joblib.dump(self.preprocessor, self.preprocessor_path)
        logger.info(f"Preprocessor saved to {self.preprocessor_path}")
        
        # Save feature statistics
        joblib.dump(self.feature_stats, self.feature_stats_path)
        logger.info(f"Feature statistics saved to {self.feature_stats_path}")
        
        # Save encoders
        joblib.dump(self.encoders, self.encoders_path)
        logger.info(f"Encoders saved to {self.encoders_path}")
        
        # Save additional metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'optimal_threshold': getattr(self, 'optimal_threshold', 0.5),
            'model_version': '2.0',
            'training_date': datetime.now().isoformat(),
            'scaler_path': self.scaler_path,
            'preprocessor_path': self.preprocessor_path
        }
        
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        logger.info(f"Metadata saved to {metadata_path}")
        
        logger.info(f"All model components saved successfully!")
    
    def load_model(self, model_path: str = None):
        """
        Load the trained model and all components
        """
        if model_path is None:
            model_path = self.model_path
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}. Using default risk scoring.")
            return False
        
        try:
            # Load model
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Load scaler explicitly
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Scaler loaded from {self.scaler_path}")
            else:
                logger.warning(f"Scaler file not found at {self.scaler_path}")
            
            # Load preprocessor (for backward compatibility)
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                logger.info(f"Preprocessor loaded from {self.preprocessor_path}")
                
                # If scaler wasn't loaded separately, use preprocessor as scaler
                if self.scaler is None:
                    self.scaler = self.preprocessor
                    logger.info("Using preprocessor as scaler for backward compatibility")
            
            # Load feature statistics
            if os.path.exists(self.feature_stats_path):
                self.feature_stats = joblib.load(self.feature_stats_path)
                logger.info(f"Feature statistics loaded from {self.feature_stats_path}")
            
            # Load encoders
            if os.path.exists(self.encoders_path):
                self.encoders = joblib.load(self.encoders_path)
                logger.info(f"Encoders loaded from {self.encoders_path}")
            
            # Load metadata
            metadata_path = model_path.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_columns = metadata['feature_columns']
                self.optimal_threshold = metadata.get('optimal_threshold', 0.5)
                logger.info(f"Metadata loaded from {metadata_path}")
            
            logger.info("All model components loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def prepare_single_transaction(self, transaction: Dict[str, Any]) -> np.ndarray:
        """
        Prepare a single transaction for prediction
        """
        # Convert transaction to DataFrame
        df = pd.DataFrame([{
            'Timestamp': pd.to_datetime(transaction.get('timestamp', datetime.now())),
            'From_Bank': transaction.get('from_bank', 0),
            'From_Account': transaction.get('from_account', 'unknown'),
            'To_Bank': transaction.get('to_bank', 0),
            'To_Account': transaction.get('to_account', 'unknown'),
            'Amount_Received': transaction.get('amount_received', transaction.get('amount', 0)),
            'Receiving_Currency': transaction.get('receiving_currency', 'USD'),
            'Amount_Paid': transaction.get('amount_paid', transaction.get('amount', 0)),
            'Payment_Currency': transaction.get('payment_currency', 'USD'),
            'Payment_Format': transaction.get('payment_format', 'Unknown')
        }])
        
        # Create features (simplified version for single transaction)
        df_featured = self.create_advanced_features(df)
        
        # Handle categorical encoding
        for col in ['Receiving_Currency', 'Payment_Currency', 'Payment_Format', 'bank_pair']:
            if col in self.encoders:
                try:
                    df_featured[f'{col}_encoded'] = self.encoders[col].transform(df_featured[col].fillna('Unknown'))
                except ValueError:
                    # Handle unknown categories
                    df_featured[f'{col}_encoded'] = 0
        
        # Fill missing features with defaults or feature stats
        for col in self.feature_columns:
            if col not in df_featured.columns:
                if col in self.feature_stats:
                    df_featured[col] = self.feature_stats[col]['median']
                else:
                    df_featured[col] = 0
        
        # Select and order features
        feature_data = df_featured[self.feature_columns].fillna(0)
        
        # Scale features using the explicit scaler
        if self.scaler is not None:
            feature_scaled = self.scaler.transform(feature_data)
            logger.debug("Features scaled using loaded scaler")
        elif self.preprocessor is not None:
            feature_scaled = self.preprocessor.transform(feature_data)
            logger.debug("Features scaled using preprocessor (backward compatibility)")
        else:
            feature_scaled = feature_data.values
            logger.warning("No scaler available, using raw features")
        
        return feature_scaled
    
    def get_basic_risk_score(self, transaction: Dict[str, Any]) -> float:
        """
        Calculate a basic risk score using rule-based approach when no trained model exists
        """
        risk_score = 0.0
        
        amount = transaction.get('amount', transaction.get('amount_paid', 0))
        
        # High amount transactions
        if amount > 10000:
            risk_score += 0.3
        elif amount > 50000:
            risk_score += 0.5
        
        # Structuring detection
        if any(abs(amount - sus_amt) <= 500 for sus_amt in self.suspicious_amounts):
            risk_score += 0.4
        
        # Round amounts
        if amount % 1000 == 0:
            risk_score += 0.1
        
        # High-risk currencies
        receiving_currency = transaction.get('receiving_currency', 'USD')
        payment_currency = transaction.get('payment_currency', 'USD')
        
        if receiving_currency in self.high_risk_currencies or payment_currency in self.high_risk_currencies:
            risk_score += 0.2
        
        # Currency mismatch
        if receiving_currency != payment_currency:
            risk_score += 0.1
        
        # Time-based risk (night transactions)
        timestamp = pd.to_datetime(transaction.get('timestamp', datetime.now()))
        hour = timestamp.hour
        
        if hour <= 6 or hour >= 22:
            risk_score += 0.1
        
        # Weekend transactions
        if timestamp.weekday() >= 5:
            risk_score += 0.05
        
        # Cap at 1.0
        return min(risk_score, 1.0)


# Global instance for API usage
_classifier = None

def get_classifier():
    """Get or create the global classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = AdvancedAMLRiskClassifier()
    return _classifier

def train_risk_classifier(df: pd.DataFrame, save_path: str = MODEL_PATH) -> Dict[str, Any]:
    """
    Train the AML risk classifier on the provided dataframe and save the model
    
    Args:
        df: DataFrame with columns [Timestamp, From_Bank, From_Account, To_Bank, To_Account, 
                                   Amount_Received, Receiving_Currency, Amount_Paid, 
                                   Payment_Currency, Payment_Format, Is_Laundering]
        save_path: Path to save the trained model
    
    Returns:
        Dictionary with training results and metrics
    """
    classifier = AdvancedAMLRiskClassifier(save_path)
    results = classifier.train_model(df)
    classifier.save_model(save_path)
    
    # Update global classifier
    global _classifier
    _classifier = classifier
    
    return results

def predict_risk_score(transaction: dict) -> float:
    """
    Predict the risk score (probability of laundering) for a given transaction
    
    Args:
        transaction: Dictionary containing transaction details
                    Required/Expected keys:
                    - amount or amount_paid: Transaction amount
                    - timestamp: Transaction timestamp (optional, defaults to now)
                    - from_bank, to_bank: Bank identifiers (optional)
                    - receiving_currency, payment_currency: Currency codes (optional)
                    - payment_format: Payment method (optional)
    
    Returns:
        Float between 0 and 1 representing risk probability
    """
    classifier = get_classifier()
    
    try:
        # Try to load model if not already loaded
        if classifier.model is None:
            model_loaded = classifier.load_model()
            
            if not model_loaded:
                # Use basic rule-based scoring if no trained model
                logger.info("Using basic rule-based risk scoring")
                return classifier.get_basic_risk_score(transaction)
        
        # Prepare transaction features
        feature_data = classifier.prepare_single_transaction(transaction)
        
        # Predict risk score
        risk_score = classifier.model.predict_proba(feature_data)[0][1]
        
        return round(float(risk_score), 4)
    
    except Exception as e:
        logger.error(f"Error predicting risk score: {str(e)}")
        # Fall back to basic scoring
        return classifier.get_basic_risk_score(transaction)

def load_training_data() -> pd.DataFrame:
    """
    Load the actual training data from the configured path
    """
    try:
        df = load_and_clean_data()
        logger.info(f"Loaded {len(df)} transactions from training data")
        logger.info(f"Laundering cases: {df['Is_Laundering'].sum()} ({df['Is_Laundering'].mean()*100:.3f}%)")
        return df
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

def classify_transaction(transaction: dict, threshold: float = None) -> dict:
    """
    Classify a transaction as suspicious or not based on risk score
    
    Args:
        transaction: Dictionary containing transaction details
        threshold: Custom threshold for classification (optional)
    
    Returns:
        Dictionary with risk_score, is_suspicious, and confidence
    """
    classifier = get_classifier()
    
    # Get risk score
    risk_score = predict_risk_score(transaction)
    
    # Use optimal threshold if available, otherwise use provided or default
    if threshold is None:
        threshold = getattr(classifier, 'optimal_threshold', 0.5)
    
    # Classify
    is_suspicious = risk_score >= threshold
    
    # Calculate confidence (distance from threshold)
    confidence = abs(risk_score - threshold) / max(threshold, 1 - threshold)
    confidence = min(confidence, 1.0)
    
    return {
        'risk_score': risk_score,
        'is_suspicious': is_suspicious,
        'confidence': round(confidence, 4),
        'threshold_used': threshold,
        'risk_level': 'HIGH' if risk_score >= 0.8 else 'MEDIUM' if risk_score >= 0.5 else 'LOW'
    }

def batch_predict(transactions: List[dict]) -> List[dict]:
    """
    Predict risk scores for multiple transactions
    
    Args:
        transactions: List of transaction dictionaries
    
    Returns:
        List of prediction results
    """
    results = []
    
    for i, transaction in enumerate(transactions):
        try:
            result = classify_transaction(transaction)
            result['transaction_id'] = i
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing transaction {i}: {str(e)}")
            results.append({
                'transaction_id': i,
                'risk_score': 0.0,
                'is_suspicious': False,
                'confidence': 0.0,
                'error': str(e)
            })
    
    return results

def get_model_info() -> dict:
    """
    Get information about the loaded model
    """
    classifier = get_classifier()
    
    if classifier.model is None:
        classifier.load_model()
    
    info = {
        'model_loaded': classifier.model is not None,
        'model_type': type(classifier.model).__name__ if classifier.model else None,
        'feature_count': len(classifier.feature_columns),
        'optimal_threshold': getattr(classifier, 'optimal_threshold', 0.5),
        'model_path': classifier.model_path,
        'scaler_available': classifier.scaler is not None,
        'encoders_count': len(classifier.encoders)
    }
    
    # Add feature importance if available
    if classifier.model and hasattr(classifier.model, 'feature_importances_'):
        feature_importance = dict(zip(classifier.feature_columns, classifier.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        info['top_features'] = top_features
    
    return info

def retrain_model(new_data_path: str = None) -> dict:
    """
    Retrain the model with new data
    
    Args:
        new_data_path: Path to new training data (optional)
    
    Returns:
        Training results
    """
    try:
        if new_data_path:
            # Load new data from specified path
            df = pd.read_csv(new_data_path)
        else:
            # Load default training data
            df = load_training_data()
        
        # Train new model
        results = train_risk_classifier(df)
        
        logger.info("Model retrained successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        raise

def validate_model(test_data: pd.DataFrame = None) -> dict:
    """
    Validate the model performance on test data
    
    Args:
        test_data: Test dataset (optional, will use holdout if not provided)
    
    Returns:
        Validation metrics
    """
    classifier = get_classifier()
    
    if classifier.model is None:
        classifier.load_model()
    
    if test_data is None:
        # Load and split data for validation
        df = load_training_data()
        _, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Is_Laundering'])
    
    # Prepare test features
    X_test, _ = classifier.prepare_features_for_training(test_data)
    y_test = test_data['Is_Laundering']
    
    # Scale features
    if classifier.scaler:
        X_test_scaled = classifier.scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Make predictions
    y_pred_proba = classifier.model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= classifier.optimal_threshold).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_score': roc_auc_score(y_test, y_pred_proba),
        'threshold': classifier.optimal_threshold,
        'test_samples': len(y_test),
        'positive_samples': y_test.sum()
    }
    
    return metrics

if __name__ == "__main__":
    """
    Main execution block for training and testing
    """
    logger.info("Starting AML Risk Classifier")
    
    try:
        # Load training data
        df = load_training_data()
        
        # Train model
        logger.info("Training model...")
        results = train_risk_classifier(df)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        
        # Test with a sample transaction
        sample_transaction = {
            'amount': 15000,
            'timestamp': datetime.now(),
            'from_bank': 123,
            'to_bank': 456,
            'receiving_currency': 'USD',
            'payment_currency': 'EUR',
            'payment_format': 'wire_transfer'
        }
        
        # Predict risk
        prediction = classify_transaction(sample_transaction)
        logger.info(f"Sample prediction: {prediction}")
        
        # Model info
        model_info = get_model_info()
        logger.info(f"Model info: {model_info}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise