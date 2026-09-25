import os
import sys
import logging

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.disable(logging.INFO)

import src.models.risk_classifier as rc  # noqa: E402
from src.models.anomaly_detector import AnomalyDetector  # noqa: E402


def make_transactions(n=3000, positive_rate=0.05, seed=0) -> pd.DataFrame:
    """Synthetic transactions in the training schema; laundering rows are large Bitcoin/ACH payments."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'Timestamp': pd.Timestamp('2022-09-01') + pd.to_timedelta(np.sort(rng.integers(0, 30 * 1440, n)), unit='min'),
        'From_Bank': rng.integers(1, 20, n),
        'From_Account': rng.choice([f'A{i}' for i in range(150)], n),
        'To_Bank': rng.integers(1, 20, n),
        'To_Account': rng.choice([f'B{i}' for i in range(150)], n),
        'Amount_Received': rng.uniform(10, 5000, n),
        'Receiving_Currency': rng.choice(['US Dollar', 'Euro'], n),
        'Amount_Paid': 0.0,
        'Payment_Currency': 'US Dollar',
        'Payment_Format': rng.choice(['Cheque', 'Credit Card', 'Wire'], n),
        'Is_Laundering': (rng.random(n) < positive_rate).astype(int),
    })
    laundering = df['Is_Laundering'] == 1
    df.loc[laundering, 'Amount_Received'] = rng.uniform(50_000, 200_000, laundering.sum())
    df.loc[laundering, 'Payment_Format'] = 'ACH'
    df['Amount_Paid'] = df['Amount_Received']
    return df


@pytest.fixture
def transactions():
    return make_transactions()


@pytest.fixture(scope="session")
def trained_classifier(tmp_path_factory):
    """Small risk classifier trained on synthetic data and saved to a temp dir,
    installed as the module-level classifier used by the API / SAR code."""
    model_dir = tmp_path_factory.mktemp("risk_model")
    clf = rc.AdvancedAMLRiskClassifier(model_path=str(model_dir / "risk_classifier_xgb.pkl"))
    clf.scaler_path = str(model_dir / "risk_classifier_scaler.pkl")
    clf.preprocessor_path = str(model_dir / "aml_preprocessor.pkl")
    clf.feature_stats_path = str(model_dir / "aml_feature_stats.pkl")
    clf.encoders_path = str(model_dir / "aml_encoders.pkl")
    clf.train_model(make_transactions())
    clf.save_model()

    previous = rc._classifier
    rc._classifier = clf
    yield clf
    rc._classifier = previous


@pytest.fixture(scope="session")
def anomaly_model_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("isolation_forest")
    detector = AnomalyDetector(model_type='isolation_forest', contamination=0.05)
    detector.train(make_transactions(n=1000).drop(columns=['Is_Laundering']))
    detector.save_model(str(model_dir))
    return str(model_dir)
