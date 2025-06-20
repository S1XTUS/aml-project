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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "risk_classifier_xgb.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "aml_preprocessor.pkl")
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
        self.feature_stats_path = FEATURE_STATS_PATH
        self.encoders_path = ENCODERS_PATH
        
        # Model components
        self.model = None
        self.preprocessor = None
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
        
        # Handle severe class imbalance with combined sampling
        # Use SMOTE to oversample minority class and RandomUnderSampler to reduce majority class
        smote = SMOTE(random_state=42, k_neighbors=min(3, y_train.sum()-1))  # Adjust k_neighbors if too few positive samples
        undersampler = RandomUnderSampler(random_state=42, sampling_strategy=0.1)  # 10% positive class
        
        # Create preprocessing pipeline
        self.preprocessor = StandardScaler()
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_test_scaled = self.preprocessor.transform(X_test)
        
        # Apply sampling
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_resampled, y_train_resampled)
        
        logger.info(f"After resampling: {len(X_train_resampled)} samples")
        logger.info(f"Positive class ratio: {y_train_resampled.mean()*100:.1f}%")
        
        # Train XGBoost model (better for imbalanced data)
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            early_stopping_rounds=50,
            n_jobs=-1
        )
        
        # Fit model
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
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        
        # Store optimal threshold
        self.optimal_threshold = optimal_threshold
        
        # Feature importance
        feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        
        results = {
            'auc_score': auc_score,
            'optimal_threshold': optimal_threshold,
            'top_features': top_features,
            'training_samples': len(X_train_resampled),
            'test_samples': len(X_test),
            'feature_count': len(feature_columns)
        }
        
        logger.info(f"Model training completed!")
        logger.info(f"AUC Score: {auc_score:.4f}")
        logger.info(f"Optimal Threshold: {optimal_threshold:.4f}")
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
        
        # Save preprocessor
        joblib.dump(self.preprocessor, self.preprocessor_path)
        
        # Save feature statistics
        joblib.dump(self.feature_stats, self.feature_stats_path)
        
        # Save encoders
        joblib.dump(self.encoders, self.encoders_path)
        
        # Save additional metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'optimal_threshold': getattr(self, 'optimal_threshold', 0.5),
            'model_version': '2.0',
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model and components saved to {model_path}")
    
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
            
            # Load preprocessor
            self.preprocessor = joblib.load(self.preprocessor_path)
            
            # Load feature statistics
            self.feature_stats = joblib.load(self.feature_stats_path)
            
            # Load encoders
            self.encoders = joblib.load(self.encoders_path)
            
            # Load metadata
            metadata_path = model_path.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_columns = metadata['feature_columns']
                self.optimal_threshold = metadata.get('optimal_threshold', 0.5)
            
            logger.info(f"Model loaded from {model_path}")
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
        
        # Scale features if preprocessor exists
        if self.preprocessor is not None:
            feature_scaled = self.preprocessor.transform(feature_data)
        else:
            feature_scaled = feature_data.values
        
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

if __name__ == "__main__":
    # Example of how to use the classifier
    
    # Sample transaction for testing
    sample_transaction = {
        'amount': 15000,
        'timestamp': '2024-06-15 14:30:00',
        'from_bank': 123,
        'to_bank': 456,
        'receiving_currency': 'USD',
        'payment_currency': 'EUR',
        'payment_format': 'wire_transfer'
    }
    
    print("Advanced AML Risk Classifier")
    print("===========================")
    print("Usage:")
    print("1. Train model: results = train_risk_classifier(your_dataframe)")
    print("2. Predict risk: score = predict_risk_score(transaction_dict)")
    print("3. Load training data: df = load_training_data()")
    
    # Test with sample prediction
    print(f"\nSample prediction (basic scoring): {predict_risk_score(sample_transaction)}")
    
    # Load actual training data
    print("\nLoading training data...")
    try:
        training_df = load_training_data()
        print(f"Loaded {len(training_df)} training transactions")
        print(f"Suspicious transactions: {training_df['Is_Laundering'].sum()} ({training_df['Is_Laundering'].mean()*100:.1f}%)")
        
        # Train model
        print("\nTraining model on actual data...")
        results = train_risk_classifier(training_df)
        print(f"Training completed successfully!")
        print(f"AUC Score: {results['auc_score']:.4f}")
        print(f"Optimal Threshold: {results['optimal_threshold']:.4f}")
        print(f"Feature Count: {results['feature_count']}")
        
        # Test prediction with trained model
        print(f"\nSample prediction (trained model): {predict_risk_score(sample_transaction)}")
        
        # Test multiple transactions
        test_transactions = [
            {
                'amount': 9000,  # Suspicious structuring amount
                'timestamp': '2024-06-15 02:30:00',  # Night transaction
                'receiving_currency': 'CHF',  # High-risk currency
                'payment_currency': 'USD',
                'payment_format': 'wire_transfer'
            },
            {
                'amount': 2500,  # Normal amount
                'timestamp': '2024-06-15 14:30:00',  # Business hours
                'receiving_currency': 'USD',
                'payment_currency': 'USD',
                'payment_format': 'card'
            },
            {
                'amount': 10000,  # Round suspicious amount
                'timestamp': '2024-06-15 23:45:00',  # Late night
                'receiving_currency': 'EUR',
                'payment_currency': 'GBP',
                'payment_format': 'cash'
            }
        ]
        
        print("\nTesting multiple transactions:")
        for i, txn in enumerate(test_transactions, 1):
            risk_score = predict_risk_score(txn)
            print(f"Transaction {i}: Risk Score = {risk_score:.4f}")
            
    except Exception as e:
        print(f"Training failed: {str(e)}")
        print("This might be due to data loading issues or insufficient suspicious transactions.")
        print("You can still use basic rule-based scoring.")
        
        # Show basic scoring works
        print(f"\nBasic scoring still works: {predict_risk_score(sample_transaction)}")
    
    print("\n" + "="*50)
    print("SETUP INSTRUCTIONS:")
    print("="*50)
    print("1. Install required packages:")
    print("   pip install pandas numpy scikit-learn imbalanced-learn xgboost joblib pyyaml")
    print("\n2. To use with your own data:")
    print("   - Ensure your DataFrame has the required columns")
    print("   - Call train_risk_classifier(your_df) to train")
    print("   - Call predict_risk_score(transaction_dict) to predict")
    print("\n3. Required DataFrame columns for training:")
    print("   - Timestamp, From_Bank, From_Account, To_Bank, To_Account")
    print("   - Amount_Received, Receiving_Currency, Amount_Paid, Payment_Currency")
    print("   - Payment_Format, Is_Laundering (target variable)")
    print("\n4. Transaction dictionary keys for prediction:")
    print("   - amount (required)")
    print("   - timestamp, from_bank, to_bank (optional)")
    print("   - receiving_currency, payment_currency, payment_format (optional)")
    
    print("\n" + "="*50)
    print("TROUBLESHOOTING:")
    print("="*50)
    print("- If you get 'Model file not found' error, this is normal on first run")
    print("- The system will use basic rule-based scoring until you train a model")
    print("- Make sure you have training data with 'Is_Laundering' column")
    print("- Check that file paths are accessible and you have write permissions")
    print("- Ensure config.yaml and data files are in the correct locations")