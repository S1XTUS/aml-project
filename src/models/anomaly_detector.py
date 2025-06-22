import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import warnings
from datetime import datetime
import logging


# Set up logging
logging.bassicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Advanced Anomaly Detection System for Financial Transactions
    Supports Isolation Forest, Autoencoder, and hybrid approaches
    """
    def __init__(self, model_type='isolation_forest', contamination=0.01, random_state=42):
        """
        Initialize the anomaly detector
        
        Args:
            model_type: 'isolation_forest', 'autoencoder', or 'hybrid'
            contamination: Expected proportion of anomalies (0.01 to 0.5)
            random_state: For reproducibility
        """
        self.model_type = model_type
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_names = []
        self.is_trained = False


    def _encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using Label Encoding
        
        Args:
            df: DataFrame with categorical features
            fit: Whether to fit encoders or transform existing data
        """
        df_encoded = df.copy()
        categorical_cols = ['From_Bank', 'To_Bank', 'Payment_Format', 'Receiving_Currency', 'Payment_Currency']

        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df[col])
                else:
                    if col in self.encoders:
                        unique_vals = self.encoders[col].classes_
                        df_encoded[col] = df[col].astype(str).apply(
                            lambda x: self.encoders[col].transform([x])[0] if x in unique_vals else -1
                        )
                        
                    else:
                        logger.warning(f"Encoder for {col} not found. Skipping encoding.")
        
        return df_encoded
                        
    def _extract_time_features(self , df):
        """Extract time-based features from timestamp"""
        df_time = df.copy()
        
        if 'Timestamp' in df.columns:
            # Extract time features
            df_time['hour'] = df_time['Timestamp'].dt.hour
            df_time['day_of_week'] = df_time['Timestamp'].dt.dayofweek
            df_time['month'] = df_time['Timestamp'].dt.month
            df_time['is_weekend'] = (df_time['Timestamp'].dt.dayofweek >= 5).astype(int)
            
            # Drop original timestamp
            df_time = df_time.drop(columns=['Timestamp'])
        
        return df_time
    

    def _create_transaction_features(self, df):
        """Create addtional transation specific features"""

        df_features = df.copy()

        if "Amount_Received" in df.columns and "Amount_Paid" in df.columns:
            df_features['amount_ratio'] = df_features['Amount_Received'] / (df_features['Amount_Paid'] + 1e-8)
            df_features['amount_difference'] = df_features['Amount_Received'] - df_features['Amount_Paid']
        
        if 'From_Account' in df.columns and 'To_Account' in df.columns:
            df_features['same_account'] = (df_features['From_Account'] == df_features['To_Account']).astype(int)
            df_features['account_diff'] = abs(df_features['From_Account'].astype(str).str.len() - 
                                            df_features['To_Account'].astype(str).str.len())
            
        # Bank similarity
        if 'From_Bank' in df.columns and 'To_Bank' in df.columns:
            df_features['same_bank'] = (df_features['From_Bank'] == df_features['To_Bank']).astype(int)

        return df_features
    
    def preprocess_data(self, df, fit=True):
        """
        Comphrensive data preprocessing pipeline
        Args:
            df: Input dataframe
            fit: Whether to fit encoders/scalers (True for training, False for prediction)
        
        Returns:
            Preprocessed feature array
        """

        logger.info("Starting data preprocessing...")

        df_processed = df.copy()
        df_processed = self._extract_time_features(df_processed)
        df_processed = self._create_transaction_features(df_processed)
        df_processed = self._encode_categorical_features(df_processed, fit=fit)

        if "Is Laundering" in df_processed.columns:
            df_processed = df_processed.drop(columns=["Is Laundering"])
        
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        X = df_processed[numeric_cols].fillna(0)

        if fit:
            self.feature_names = numeric_cols
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            logger.info(f"Fitted scaler on {len(numeric_cols)} features: {numeric_cols}")

        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Please train the model first")
            
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing Features in prediction data : {missing_features}, adding with zero values")
                for feature in missing_features:
                    X[feature] = 0
            X = X[self.feature_names]
            X_scaled = self.scaler.transform(X)
        logger.info(f"Proprocessing Complete. Shape : {X_scaled.shape}")
        return X_scaled
    
    def _train_isolation_forest(self, X):
        """Train Isolation Forest model"""
        logger.info("Training Isolation Forest...")
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X)
        logger.info("Isolation Forest training complete.")

    
    def _train_autoencoder(self, X):
        """Train Autoencoder for anomaly detection"""
        logger.info("Training Autoencoder...")
        
        # Simple autoencoder using MLPRegressor
        input_dim = X.shape[1]
        hidden_dim = max(2, input_dim // 2)  # Bottleneck layer
        
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_dim, hidden_dim//2, hidden_dim),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Train autoencoder to reconstruct input
        self.model.fit(X, X)
        
        # Calculate reconstruction error threshold
        reconstructed = self.model.predict(X)
        reconstruction_errors = np.mean((X - reconstructed) ** 2, axis=1)
        self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
        
        logger.info(f"Autoencoder training complete. Threshold: {self.threshold:.4f}")
    
    def train(self, df):
        """
        Train the anomaly detection model
        
        Args:
            df: Training dataframe with transaction data
        """
        logger.info(f"Training anomaly detector with {self.model_type} algorithm...")
        
        # Preprocess data
        X = self.preprocess_data(df, fit=True)
        
        # Train based on model type
        if self.model_type == 'isolation_forest':
            self._train_isolation_forest(X)
        elif self.model_type == 'autoencoder':
            self._train_autoencoder(X)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.is_trained = True
        logger.info("Model training completed successfully!")

    
    def predict_anomalies(self, df):
        """
        Predict anomalies in new data
        
        Args:
            df: Dataframe with transaction data to analyze
            
        Returns:
            Dataframe with anomaly predictions and scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        logger.info(f"Predicting anomalies for {len(df)} transactions...")
        
        # Preprocess data
        X = self.preprocess_data(df, fit=False)
        
        # Make predictions based on model type
        if self.model_type == 'isolation_forest':
            # Predict: -1 is anomaly, 1 is normal
            predictions = self.model.predict(X)
            anomaly_scores = self.model.decision_function(X)
            anomaly_flags = (predictions == -1).astype(int)
            
        elif self.model_type == 'autoencoder':
            # Calculate reconstruction error
            reconstructed = self.model.predict(X)
            anomaly_scores = np.mean((X - reconstructed) ** 2, axis=1)
            anomaly_flags = (anomaly_scores > self.threshold).astype(int)
            # Invert scores for consistency (higher = more anomalous)
            anomaly_scores = -anomaly_scores
        
        # Add results to dataframe
        df_result = df.copy()
        df_result['anomaly_flag'] = anomaly_flags
        df_result['anomaly_score'] = anomaly_scores
        df_result['risk_level'] = pd.cut(
            anomaly_scores, 
            bins=[-np.inf, np.percentile(anomaly_scores, 25), 
                  np.percentile(anomaly_scores, 75), np.inf],
            labels=['High Risk', 'Medium Risk', 'Low Risk']
        )
        
        anomaly_count = anomaly_flags.sum()
        logger.info(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")
        
        return df_result
    
    def evaluate(self, df_test):
        """
        Evaluate model performance if ground truth labels are available
        
        Args:
            df_test: Test dataframe with 'Is_Laundering' column
        """
        if 'Is_Laundering' not in df_test.columns:
            logger.warning("No ground truth labels found. Skipping evaluation.")
            return None
        
        # Get predictions
        df_pred = self.predict_anomalies(df_test)
        
        # Calculate metrics
        y_true = df_test['Is_Laundering']
        y_pred = df_pred['anomaly_flag']
        
        print("\n=== MODEL EVALUATION ===")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        # Additional metrics
        precision = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
        recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nPrecision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def save_model(self, model_dir="models"):
        """Save trained model and preprocessors"""
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model components
        model_path = os.path.join(model_dir, f"anomaly_detector_{self.model_type}.joblib")
        scaler_path = os.path.join(model_dir, "anomaly_scaler.joblib")
        encoders_path = os.path.join(model_dir, "anomaly_encoders.joblib")
        config_path = os.path.join(model_dir, "anomaly_config.joblib")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.encoders, encoders_path)
        
        # Save configuration
        config = {
            'model_type': self.model_type,
            'contamination': self.contamination,
            'feature_names': self.feature_names,
            'threshold': getattr(self, 'threshold', None)
        }
        joblib.dump(config, config_path)
        
        logger.info(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir="models"):
        """Load trained model and preprocessors"""
        try:
            # Load configuration
            config_path = os.path.join(model_dir, "anomaly_config.joblib")
            config = joblib.load(config_path)
            
            self.model_type = config['model_type']
            self.contamination = config['contamination']
            self.feature_names = config['feature_names']
            if 'threshold' in config:
                self.threshold = config['threshold']
            
            # Load model components
            model_path = os.path.join(model_dir, f"anomaly_detector_{self.model_type}.joblib")
            scaler_path = os.path.join(model_dir, "anomaly_scaler.joblib")
            encoders_path = os.path.join(model_dir, "anomaly_encoders.joblib")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.encoders = joblib.load(encoders_path)
            
            self.is_trained = True
            logger.info(f"Model loaded from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


    # Convenience functions for backward compatibility and easy usage
    def load_data(path):
        """Load transaction data from CSV"""
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} transactions from {path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def train_anomaly_model(data_path=None, model_type='isolation_forest', contamination=0.05, model_dir="models"):
        """
        Train and save anomaly detection model
        
        Args:
            data_path: Path to training data CSV
            model_type: 'isolation_forest' or 'autoencoder'
            contamination: Expected proportion of anomalies
            model_dir: Directory to save model
        """
        # Default path if not provided
        if data_path is None:
            data_path = r"D:\aml-project\data\processed\LI-Small_Trans.csv"
        
        # Load data
        df = load_data(data_path)
        
        # Initialize and train detector
        detector = AnomalyDetector(model_type=model_type, contamination=contamination)
        detector.train(df)
        
        # Evaluate if labels available
        if 'Is_Laundering' in df.columns:
            # Split for evaluation
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            detector.train(train_df)
            detector.evaluate(test_df)
        
        # Save model
        detector.save_model(model_dir)
        print(f"✅ {model_type.title()} anomaly detection model trained and saved to {model_dir}")
        
        return detector

    def predict_anomalies(df_new, model_dir="models"):
        """
        Predict anomalies using saved model
        
        Args:
            df_new: New transaction data to analyze
            model_dir: Directory containing saved model
            
        Returns:
            DataFrame with anomaly predictions
        """
        # Load model
        detector = AnomalyDetector()
        detector.load_model(model_dir)
        
        # Make predictions
        result = detector.predict_anomalies(df_new)
        
        return result

    def analyze_anomalies(df_results):
        """Analyze and summarize anomaly detection results"""
        if 'anomaly_flag' not in df_results.columns:
            logger.error("No anomaly predictions found in data.")
            return
        
        total_transactions = len(df_results)
        anomaly_count = df_results['anomaly_flag'].sum()
        anomaly_rate = (anomaly_count / total_transactions) * 100
        
        print(f"\n=== ANOMALY ANALYSIS SUMMARY ===")
        print(f"Total Transactions: {total_transactions:,}")
        print(f"Anomalies Detected: {anomaly_count:,} ({anomaly_rate:.2f}%)")
        
        if 'risk_level' in df_results.columns:
            print(f"\nRisk Level Distribution:")
            risk_dist = df_results['risk_level'].value_counts()
            for level, count in risk_dist.items():
                print(f"  {level}: {count:,} ({count/total_transactions*100:.1f}%)")
        
        # Show top anomalies
        if 'anomaly_score' in df_results.columns and anomaly_count > 0:
            print(f"\nTop 5 Most Anomalous Transactions:")
            top_anomalies = df_results[df_results['anomaly_flag'] == 1].nlargest(5, 'anomaly_score')
            for idx, row in top_anomalies.iterrows():
                print(f"  Transaction {idx}: Score={row['anomaly_score']:.3f}, Amount=${row.get('Amount_Received', 'N/A')}")


    # Main execution
    if __name__ == "__main__":
        # Configuration
        DATA_PATH = r"D:\aml-project\data\processed\LI-Small_Trans.csv"
        MODEL_DIR = "models"
        
        # Train models with different algorithms
        print("Training Isolation Forest model...")
        detector_if = train_anomaly_model(
            data_path=DATA_PATH,
            model_type='isolation_forest',
            contamination=0.05,
            model_dir=MODEL_DIR
        )
        
        print("\nTraining Autoencoder model...")
        detector_ae = train_anomaly_model(
            data_path=DATA_PATH,
            model_type='autoencoder',
            contamination=0.05,
            model_dir="models_autoencoder"
        )
        
        # Example prediction usage
        print("\n" + "="*50)
        print("EXAMPLE: Predicting on new data")
        
        # Load some sample data for prediction
        sample_data = load_data(DATA_PATH).head(100)  # Use first 100 rows as example
        
        # Predict with Isolation Forest
        results = predict_anomalies(sample_data, MODEL_DIR)
        analyze_anomalies(results)
        
        print("\n✅ Anomaly detection system ready for use!")
        print("\nUsage Examples:")
        print("  # Train new model:")
        print("  detector = train_anomaly_model('your_data.csv', 'isolation_forest')")
        print("  # Predict anomalies:")
        print("  results = predict_anomalies(your_df, 'models')")
        print("  # Analyze results:")
        print("  analyze_anomalies(results)")