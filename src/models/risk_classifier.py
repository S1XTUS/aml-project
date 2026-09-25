import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.load_data import load_and_clean_data, standardize_columns
from src.utils.config_loader import risk_level_from_score
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, accuracy_score)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the project root directory (go up two levels from src/models/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "risk_classifier_xgb.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "aml_preprocessor.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "risk_classifier_scaler.pkl")  # Explicit scaler path
FEATURE_STATS_PATH = os.path.join(MODELS_DIR, "aml_feature_stats.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "aml_encoders.pkl")

CATEGORICAL_COLUMNS = ['Receiving_Currency', 'Payment_Currency', 'Payment_Format', 'bank_pair']
EXCLUDE_COLUMNS = ['Timestamp', 'From_Account', 'To_Account', 'Receiving_Currency',
                   'Payment_Currency', 'Payment_Format', 'bank_pair', 'Is_Laundering']
VELOCITY_WINDOWS = ['1D', '7D', '30D']

# The training data (IBM AML) spells currencies and payment formats out in full.
# API / dashboard callers use ISO codes and snake_case, so normalise them before scoring.
CURRENCY_ALIASES = {
    'USD': 'US Dollar', 'EUR': 'Euro', 'GBP': 'UK Pound', 'CHF': 'Swiss Franc', 'JPY': 'Yen',
    'CAD': 'Canadian Dollar', 'AUD': 'Australian Dollar', 'CNY': 'Yuan', 'INR': 'Rupee',
    'RUB': 'Ruble', 'BRL': 'Brazil Real', 'MXN': 'Mexican Peso', 'SAR': 'Saudi Riyal',
    'ILS': 'Shekel', 'BTC': 'Bitcoin'
}
PAYMENT_FORMAT_ALIASES = {
    'wire': 'Wire', 'wire_transfer': 'Wire', 'international_transfer': 'Wire',
    'ach': 'ACH', 'check': 'Cheque', 'cheque': 'Cheque', 'cash': 'Cash', 'cash_deposit': 'Cash',
    'card': 'Credit Card', 'credit_card': 'Credit Card', 'credit card': 'Credit Card',
    'cash_withdrawal': 'Cash', 'debit_card': 'Credit Card',
    'crypto': 'Bitcoin', 'bitcoin': 'Bitcoin', 'reinvestment': 'Reinvestment'
}
UNKNOWN_BANK = -1


def normalize_currency(value: Any) -> str:
    value = str(value).strip()
    return CURRENCY_ALIASES.get(value.upper(), value)


def normalize_payment_format(value: Any) -> str:
    value = str(value).strip()
    return PAYMENT_FORMAT_ALIASES.get(value.lower(), value)


def normalize_bank(value: Any) -> int:
    """Training data uses numeric bank IDs; bank names are unknown to the model."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return UNKNOWN_BANK


def _trailing_window_stats(accounts: pd.Series, timestamps: pd.Series, amounts: pd.Series,
                           window: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-row count and sum of the same account's transactions in (t - window, t],
    up to and including the row itself. Rows must be sorted by (account, timestamp).

    Equivalent to groupby(account).rolling(window, on=timestamp) but vectorised:
    each row's (account, time) is packed into one sortable int64 key, and the
    window start is found with a single searchsorted.
    """
    group_ids = pd.factorize(accounts, sort=False)[0].astype(np.int64)
    seconds = (timestamps.values.astype('datetime64[s]').astype(np.int64))
    seconds = seconds - seconds.min()
    window_seconds = int(pd.Timedelta(window).total_seconds())
    stride = int(seconds.max()) + window_seconds + 1  # keeps windows inside their account block
    keys = group_ids * stride + seconds

    positions = np.arange(len(keys))
    starts = np.searchsorted(keys, keys - window_seconds, side='right')
    count = (positions - starts + 1).astype(float)

    cumulative = np.concatenate([[0.0], np.cumsum(amounts.to_numpy(dtype=float))])
    total = cumulative[positions + 1] - cumulative[starts]
    return count, total


def transaction_from_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Map an API / dashboard case payload to the transaction dict used for scoring."""
    return {
        'case_id': case.get('case_id'),
        'timestamp': case.get('transaction_time'),
        'from_bank': case.get('from_bank'),
        'from_account': case.get('from_account'),
        'to_bank': case.get('to_bank'),
        'to_account': case.get('to_account'),
        'amount': case.get('amount'),
        'amount_paid': case.get('amount'),
        'amount_received': case.get('amount'),
        'receiving_currency': case.get('currency', 'USD'),
        'payment_currency': case.get('currency', 'USD'),
        'payment_format': case.get('transaction_type', 'Unknown'),
    }


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
        self.rare_payment_formats = set()

        # High-risk patterns (values as they appear in the training data)
        self.high_risk_currencies = {'Bitcoin'}
        self.suspicious_amounts = [10000, 9000, 8000, 5000]  # Common structuring amounts

    def create_advanced_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Create features from the transaction data.

        Rows are returned in the same order and with the same index as `df`, so labels
        taken from `df` stay aligned. Account-history features only look at *earlier*
        transactions of the same account, so they can be computed identically at
        training time and for a single live transaction (which then looks like an
        account's first transaction).
        """
        logger.info("Creating advanced features...")
        original_index = df.index
        df_featured = df.reset_index(drop=True)

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
        amount_paid = df_featured['Amount_Paid']
        df_featured['amount_ratio'] = df_featured['Amount_Received'] / (amount_paid + 1e-6)
        df_featured['amount_difference'] = (df_featured['Amount_Received'] - amount_paid).abs()
        df_featured['amount_difference_pct'] = df_featured['amount_difference'] / (amount_paid + 1e-6)

        # Currency features
        df_featured['currency_match'] = (df_featured['Receiving_Currency'] == df_featured['Payment_Currency']).astype(int)
        df_featured['involves_high_risk_currency'] = (
            df_featured['Receiving_Currency'].isin(self.high_risk_currencies) |
            df_featured['Payment_Currency'].isin(self.high_risk_currencies)
        ).astype(int)

        # Structuring detection (amounts close to reporting thresholds)
        near_threshold = np.zeros(len(df_featured), dtype=bool)
        for amt in self.suspicious_amounts:
            near_threshold |= (amount_paid - amt).abs().values <= 500
        df_featured['potential_structuring'] = near_threshold.astype(int)

        # Round number detection
        df_featured['is_round_amount'] = ((amount_paid % 1000) == 0).astype(int)

        # Bank relationship features
        df_featured['same_bank'] = (df_featured['From_Bank'] == df_featured['To_Bank']).astype(int)
        df_featured['bank_pair'] = df_featured['From_Bank'].astype(str) + '_' + df_featured['To_Bank'].astype(str)

        # --- Account history features (past-only) ---
        df_featured = df_featured.sort_values(['From_Account', 'Timestamp'], kind='mergesort')
        by_sender = df_featured.groupby('From_Account', sort=False)

        # Velocity: transactions in trailing windows up to and including this one
        for window in VELOCITY_WINDOWS:
            count, total = _trailing_window_stats(
                df_featured['From_Account'], df_featured['Timestamp'], df_featured['Amount_Paid'], window)
            df_featured[f'from_account_tx_count_{window}'] = count
            df_featured[f'from_account_tx_sum_{window}'] = total

        # Running statistics over the account's *previous* transactions
        prev_count = by_sender.cumcount()
        with np.errstate(divide='ignore', invalid='ignore'):
            for col, name in [('Amount_Paid', 'Amount_Paid'), ('hour', 'hour'), ('same_bank', 'same_bank')]:
                values = df_featured[col].astype(float)
                prev_sum = by_sender[col].cumsum() - values
                df_featured[f'from_account_{name}_mean'] = (prev_sum / prev_count).where(prev_count > 0)
                if col == 'Amount_Paid':
                    prev_sq = (values ** 2).groupby(df_featured['From_Account'], sort=False).cumsum() - values ** 2
                    var = (prev_sq / prev_count - (prev_sum / prev_count) ** 2).clip(lower=0)
                    df_featured['from_account_Amount_Paid_std'] = np.sqrt(var).where(prev_count > 1)
        df_featured['from_account_Amount_Paid_count'] = prev_count

        df_featured['amount_zscore'] = (
            (amount_paid - df_featured['from_account_Amount_Paid_mean']) /
            (df_featured['from_account_Amount_Paid_std'] + 1e-6)
        ).fillna(0)
        df_featured['hour_deviation'] = (df_featured['hour'] - df_featured['from_account_hour_mean']).abs().fillna(0)

        # Network features: distinct counterparties seen so far
        new_recipient = ~df_featured.duplicated(['From_Account', 'To_Account'])
        df_featured['unique_recipients_count'] = new_recipient.groupby(df_featured['From_Account'], sort=False).cumsum()

        df_featured = df_featured.sort_values(['To_Account', 'Timestamp'], kind='mergesort')
        for window in VELOCITY_WINDOWS:
            count, _ = _trailing_window_stats(
                df_featured['To_Account'], df_featured['Timestamp'], df_featured['Amount_Received'], window)
            df_featured[f'to_account_tx_count_{window}'] = count
        new_sender = ~df_featured.duplicated(['To_Account', 'From_Account'])
        df_featured['unique_senders_count'] = new_sender.groupby(df_featured['To_Account'], sort=False).cumsum()

        # Payment format rarity is learned from the training data only
        if fit:
            format_share = df_featured['Payment_Format'].value_counts(normalize=True)
            self.rare_payment_formats = set(format_share[format_share < 0.01].index)
        df_featured['rare_payment_format'] = df_featured['Payment_Format'].isin(self.rare_payment_formats).astype(int)

        # Restore the caller's row order and index
        df_featured = df_featured.sort_index()
        df_featured.index = original_index

        logger.info(f"Created {len(df_featured.columns) - len(df.columns)} new features")
        return df_featured

    def _encode_categoricals(self, df_featured: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_COLUMNS:
            if col not in df_featured.columns:
                continue
            values = df_featured[col].fillna('Unknown').astype(str)
            if fit:
                encoder = LabelEncoder()
                df_featured[f'{col}_encoded'] = encoder.fit_transform(values)
                self.encoders[col] = encoder
            elif col in self.encoders:
                mapping = {str(c): i for i, c in enumerate(self.encoders[col].classes_)}
                df_featured[f'{col}_encoded'] = values.map(mapping).fillna(-1).astype(int)
        return df_featured

    def prepare_features_for_training(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Build the model feature matrix, row-aligned with `df`.

        fit=True learns encoders, feature columns and statistics (training);
        fit=False reuses the fitted ones (validation / scoring).
        """
        logger.info("Preparing features...")

        df_featured = self.create_advanced_features(df, fit=fit)
        df_featured = self._encode_categoricals(df_featured, fit=fit)

        if fit:
            self.feature_columns = [col for col in df_featured.columns if col not in EXCLUDE_COLUMNS]

        X = df_featured.reindex(columns=self.feature_columns).fillna(0)

        if fit:
            self.feature_stats = {
                col: {
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'min': X[col].min(),
                    'max': X[col].max(),
                    'median': X[col].median()
                } for col in self.feature_columns
            }

        logger.info(f"Prepared {len(self.feature_columns)} features")
        return X, self.feature_columns

    def train_model(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.15) -> Dict[str, Any]:
        """
        Train the AML risk classification model.

        The data is split into train / validation / test. Early stopping and the
        decision threshold use the validation set; reported metrics use the untouched test set.
        """
        logger.info(f"Starting model training on {len(df)} transactions...")
        logger.info(f"Laundering cases: {df['Is_Laundering'].sum()} ({df['Is_Laundering'].mean()*100:.3f}%)")

        X, feature_columns = self.prepare_features_for_training(df, fit=True)
        y = df['Is_Laundering'].astype(int)
        assert X.index.equals(y.index), "Feature rows are not aligned with labels"

        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=test_size, random_state=42, stratify=y.values
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size, random_state=42, stratify=y.values[train_idx]
        )
        X_train, y_train = X.iloc[train_idx], y.values[train_idx]
        X_val, y_val = X.iloc[val_idx], y.values[val_idx]
        X_test, y_test = X.iloc[test_idx], y.values[test_idx]

        logger.info(f"Train / validation / test: {len(X_train)} / {len(X_val)} / {len(X_test)}")

        self.scaler = StandardScaler()
        self.preprocessor = self.scaler  # Keep backward compatibility
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = self.scaler.transform(X_val).astype(np.float32)
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32)

        # Class imbalance is handled with class weights only (no synthetic oversampling)
        positive_count = int(y_train.sum())
        negative_count = len(y_train) - positive_count
        scale_pos_weight = negative_count / max(positive_count, 1)
        logger.info(f"Class distribution - Positive: {positive_count}, Negative: {negative_count}")

        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42,
            eval_metric='aucpr',
            early_stopping_rounds=30,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight
        )

        logger.info(f"Training XGBoost with scale_pos_weight={scale_pos_weight:.2f}")
        self.model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

        # Pick the threshold on validation data, optimising F2 (recall-weighted)
        val_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
        f2_scores = 5 * (precision * recall) / (4 * precision + recall + 1e-8)
        self.optimal_threshold = float(thresholds[np.argmax(f2_scores[:-1])])

        # Evaluate on the held-out test set
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred_optimal = (y_pred_proba >= self.optimal_threshold).astype(int)

        auc_score = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        precision_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
        recall_optimal = recall_score(y_test, y_pred_optimal, zero_division=0)
        f1_optimal = f1_score(y_test, y_pred_optimal, zero_division=0)

        feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]

        results = {
            'auc_score': auc_score,
            'pr_auc': pr_auc,
            'optimal_threshold': self.optimal_threshold,
            'precision': precision_optimal,
            'recall': recall_optimal,
            'f1_score': f1_optimal,
            'best_iteration': self.model.best_iteration,
            'top_features': top_features,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_count': len(feature_columns),
            'original_positive_ratio': positive_count / len(y_train),
            'scale_pos_weight': scale_pos_weight
        }

        logger.info("Model training completed!")
        logger.info(f"Test ROC AUC: {auc_score:.4f} | PR AUC: {pr_auc:.4f} (baseline {y_test.mean():.5f})")
        logger.info(f"Threshold (from validation): {self.optimal_threshold:.4f}")
        logger.info(f"Test Precision: {precision_optimal:.4f} | Recall: {recall_optimal:.4f} | F1: {f1_optimal:.4f}")
        logger.info(f"Top 5 features: {[f[0] for f in top_features[:5]]}")

        return results

    def save_model(self, model_path: str = None):
        """
        Save the trained model and all components
        """
        if model_path is None:
            model_path = self.model_path

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Scaler saved to {self.scaler_path}")

        # Save preprocessor (for backward compatibility)
        joblib.dump(self.preprocessor, self.preprocessor_path)

        joblib.dump(self.feature_stats, self.feature_stats_path)
        logger.info(f"Feature statistics saved to {self.feature_stats_path}")

        joblib.dump(self.encoders, self.encoders_path)
        logger.info(f"Encoders saved to {self.encoders_path}")

        metadata = {
            'feature_columns': self.feature_columns,
            'optimal_threshold': self.optimal_threshold,
            'rare_payment_formats': sorted(self.rare_payment_formats),
            'high_risk_currencies': sorted(self.high_risk_currencies),
            'model_version': '3.0',
            'training_date': datetime.now().isoformat(),
            'scaler_path': self.scaler_path,
            'preprocessor_path': self.preprocessor_path
        }

        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        logger.info(f"Metadata saved to {metadata_path}")

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
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            elif os.path.exists(self.preprocessor_path):
                self.scaler = joblib.load(self.preprocessor_path)
            else:
                logger.warning(f"Scaler file not found at {self.scaler_path}")
            self.preprocessor = self.scaler

            if os.path.exists(self.feature_stats_path):
                self.feature_stats = joblib.load(self.feature_stats_path)

            if os.path.exists(self.encoders_path):
                self.encoders = joblib.load(self.encoders_path)

            metadata_path = model_path.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_columns = metadata['feature_columns']
                self.optimal_threshold = metadata.get('optimal_threshold', 0.5)
                self.rare_payment_formats = set(metadata.get('rare_payment_formats', []))
                if 'high_risk_currencies' in metadata:
                    self.high_risk_currencies = set(metadata['high_risk_currencies'])

            n_model_features = getattr(self.model, 'n_features_in_', len(self.feature_columns))
            if n_model_features != len(self.feature_columns):
                raise ValueError(f"Model expects {n_model_features} features but metadata lists "
                                 f"{len(self.feature_columns)}; retrain the model.")

            logger.info("All model components loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            return False

    def _transaction_frame(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """Turn an API-style transaction dict into a one-row frame in training schema."""
        return pd.DataFrame([{
            'Timestamp': pd.to_datetime(transaction.get('timestamp') or datetime.now()),
            'From_Bank': normalize_bank(transaction.get('from_bank')),
            'From_Account': str(transaction.get('from_account') or 'unknown'),
            'To_Bank': normalize_bank(transaction.get('to_bank')),
            'To_Account': str(transaction.get('to_account') or 'unknown'),
            'Amount_Received': float(transaction.get('amount_received', transaction.get('amount', 0)) or 0),
            'Receiving_Currency': normalize_currency(transaction.get('receiving_currency', 'USD')),
            'Amount_Paid': float(transaction.get('amount_paid', transaction.get('amount', 0)) or 0),
            'Payment_Currency': normalize_currency(transaction.get('payment_currency', 'USD')),
            'Payment_Format': normalize_payment_format(transaction.get('payment_format', 'Unknown'))
        }])

    def build_feature_frame(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """
        Unscaled model features for a single transaction, using the same code path as training.
        """
        df = self._transaction_frame(transaction)
        X, _ = self.prepare_features_for_training(df, fit=False)

        # Bank names are mapped to UNKNOWN_BANK, so compare the raw values for same_bank
        if 'same_bank' in X.columns:
            X['same_bank'] = int(str(transaction.get('from_bank')) == str(transaction.get('to_bank')))
        return X

    def prepare_single_transaction(self, transaction: Dict[str, Any]) -> np.ndarray:
        """
        Prepare a single transaction for prediction (scaled feature vector)
        """
        feature_data = self.build_feature_frame(transaction)
        if self.scaler is None:
            logger.warning("No scaler available, using raw features")
            return feature_data.values
        return self.scaler.transform(feature_data)

    def get_basic_risk_score(self, transaction: Dict[str, Any]) -> float:
        """
        Calculate a basic risk score using rule-based approach when no trained model exists
        """
        risk_score = 0.0

        amount = float(transaction.get('amount', transaction.get('amount_paid', 0)) or 0)

        # High amount transactions
        if amount > 50000:
            risk_score += 0.5
        elif amount > 10000:
            risk_score += 0.3

        # Structuring detection
        if any(abs(amount - sus_amt) <= 500 for sus_amt in self.suspicious_amounts):
            risk_score += 0.4

        # Round amounts
        if amount > 0 and amount % 1000 == 0:
            risk_score += 0.1

        receiving_currency = normalize_currency(transaction.get('receiving_currency', 'USD'))
        payment_currency = normalize_currency(transaction.get('payment_currency', 'USD'))

        if receiving_currency in self.high_risk_currencies or payment_currency in self.high_risk_currencies:
            risk_score += 0.2

        # Currency mismatch
        if receiving_currency != payment_currency:
            risk_score += 0.1

        # Time-based risk (night transactions)
        timestamp = pd.to_datetime(transaction.get('timestamp') or datetime.now())
        if timestamp.hour <= 6 or timestamp.hour >= 22:
            risk_score += 0.1

        # Weekend transactions
        if timestamp.weekday() >= 5:
            risk_score += 0.05

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

def score_transaction(transaction: dict) -> Tuple[float, str]:
    """
    Risk score for a transaction plus the method used: 'model' or 'rules'
    (rules are used when no trained model is available or scoring fails).
    """
    classifier = get_classifier()

    if classifier.model is None and not classifier.load_model():
        logger.warning("No trained model available - using rule-based risk scoring")
        return classifier.get_basic_risk_score(transaction), 'rules'

    try:
        feature_data = classifier.prepare_single_transaction(transaction)
        risk_score = classifier.model.predict_proba(feature_data)[0][1]
        return round(float(risk_score), 4), 'model'
    except Exception as e:
        logger.error(f"Model scoring failed, falling back to rules: {str(e)}")
        return classifier.get_basic_risk_score(transaction), 'rules'

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
    return score_transaction(transaction)[0]

def load_training_data() -> pd.DataFrame:
    """
    Load the actual training data from the configured path
    """
    df = load_and_clean_data()
    logger.info(f"Loaded {len(df)} transactions from training data")
    logger.info(f"Laundering cases: {df['Is_Laundering'].sum()} ({df['Is_Laundering'].mean()*100:.3f}%)")
    return df

def classify_transaction(transaction: dict, threshold: float = None) -> dict:
    """
    Classify a transaction as suspicious or not based on risk score

    Args:
        transaction: Dictionary containing transaction details
        threshold: Custom threshold for classification (optional)

    Returns:
        Dictionary with risk_score, is_suspicious, confidence, risk_level and scoring_method
    """
    classifier = get_classifier()

    risk_score, method = score_transaction(transaction)

    # The tuned threshold only applies to model probabilities
    if threshold is None:
        threshold = classifier.optimal_threshold if method == 'model' else 0.5

    is_suspicious = risk_score >= threshold

    # Calculate confidence (distance from threshold)
    confidence = abs(risk_score - threshold) / max(threshold, 1 - threshold)
    confidence = min(confidence, 1.0)

    return {
        'risk_score': risk_score,
        'is_suspicious': bool(is_suspicious),
        'confidence': round(confidence, 4),
        'threshold_used': threshold,
        'risk_level': risk_level_from_score(risk_score),
        'scoring_method': method
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
        'optimal_threshold': classifier.optimal_threshold,
        'model_path': classifier.model_path,
        'scaler_available': classifier.scaler is not None,
        'encoders_count': len(classifier.encoders)
    }

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
            df = standardize_columns(pd.read_csv(new_data_path))
        else:
            df = load_training_data()

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
        test_data: Test dataset (optional; defaults to the same held-out test split used in training)

    Returns:
        Validation metrics
    """
    classifier = get_classifier()

    if classifier.model is None and not classifier.load_model():
        raise RuntimeError("No trained model available to validate")

    if test_data is None:
        # Features need each account's full history, so build them on all data
        # and then select the same test rows train_model() held out.
        df = load_training_data()
        X_all, _ = classifier.prepare_features_for_training(df, fit=False)
        y_all = df['Is_Laundering'].astype(int).values
        _, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y_all)
        X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]
    else:
        X_test, _ = classifier.prepare_features_for_training(test_data, fit=False)
        y_test = test_data['Is_Laundering'].astype(int).values

    X_test_scaled = classifier.scaler.transform(X_test) if classifier.scaler else X_test

    y_pred_proba = classifier.model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= classifier.optimal_threshold).astype(int)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc_score': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'threshold': classifier.optimal_threshold,
        'test_samples': len(y_test),
        'positive_samples': int(y_test.sum())
    }

if __name__ == "__main__":
    """
    Main execution block for training and testing
    """
    logger.info("Starting AML Risk Classifier")

    try:
        df = load_training_data()

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

        prediction = classify_transaction(sample_transaction)
        logger.info(f"Sample prediction: {prediction}")

        model_info = get_model_info()
        logger.info(f"Model info: {model_info}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
