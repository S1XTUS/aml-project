import numpy as np
import pandas as pd
import pytest

import src.models.risk_classifier as rc
from src.models.anomaly_detector import AnomalyDetector
from tests.conftest import make_transactions


def test_features_stay_aligned_with_labels(transactions):
    """Regression: feature engineering used to reorder rows, pairing features with the wrong labels."""
    shuffled = transactions.sample(frac=1, random_state=1)
    shuffled.index = shuffled.index + 10_000
    X, _ = rc.AdvancedAMLRiskClassifier('__none__.pkl').prepare_features_for_training(shuffled)

    assert X.index.equals(shuffled.index)
    np.testing.assert_allclose(X['Amount_Paid'].values, shuffled['Amount_Paid'].values)


def test_account_history_uses_only_past_transactions(transactions):
    clf = rc.AdvancedAMLRiskClassifier('__none__.pkl')
    base, _ = clf.prepare_features_for_training(transactions)

    future = transactions.copy()
    future['Timestamp'] = future['Timestamp'] + pd.Timedelta(days=365)
    future.index = future.index + len(transactions)
    extended, _ = clf.prepare_features_for_training(pd.concat([transactions, future]))

    history_cols = [c for c in base.columns if c.startswith(('from_account_', 'to_account_', 'unique_'))
                    or c in ('amount_zscore', 'hour_deviation')]
    pd.testing.assert_frame_equal(base[history_cols], extended.loc[base.index, history_cols])


def test_first_transaction_has_no_history(transactions):
    X, _ = rc.AdvancedAMLRiskClassifier('__none__.pkl').prepare_features_for_training(transactions)
    first = transactions.sort_values('Timestamp').groupby('From_Account').head(1).index
    assert (X.loc[first, 'from_account_Amount_Paid_count'] == 0).all()
    assert (X.loc[first, 'from_account_tx_count_1D'] == 1).all()


@pytest.mark.parametrize("window", rc.VELOCITY_WINDOWS)
def test_trailing_window_matches_pandas_rolling(window):
    df = make_transactions(n=2000).sort_values(['From_Account', 'Timestamp'], kind='mergesort')
    df['Timestamp'] = df['Timestamp'].dt.floor('6h')  # force timestamp ties
    expected = df.groupby('From_Account', sort=False).rolling(window, on='Timestamp')['Amount_Paid']

    count, total = rc._trailing_window_stats(df['From_Account'], df['Timestamp'], df['Amount_Paid'], window)

    np.testing.assert_array_equal(count, expected.count().to_numpy())
    np.testing.assert_allclose(total, expected.sum().to_numpy())


def test_serving_normalizes_api_values():
    assert rc.normalize_currency('USD') == 'US Dollar'
    assert rc.normalize_currency('Euro') == 'Euro'
    assert rc.normalize_payment_format('wire_transfer') == 'Wire'
    assert rc.normalize_payment_format('ACH') == 'ACH'
    assert rc.normalize_bank('070') == 70
    assert rc.normalize_bank('Alpha Bank') == rc.UNKNOWN_BANK


def test_training_reports_held_out_metrics(trained_classifier):
    assert 0 < trained_classifier.optimal_threshold < 1
    assert len(trained_classifier.feature_columns) == trained_classifier.model.n_features_in_


def test_dashboard_payload_is_scored_by_model(trained_classifier):
    """Regression: bank names made the model fail and silently fall back to rules."""
    tx = rc.transaction_from_case({
        'transaction_time': '2024-01-01T10:00:00Z', 'from_bank': 'Alpha Bank', 'to_bank': 'Beta Bank',
        'from_account': '1', 'to_account': '2', 'amount': 1000.0, 'currency': 'USD',
        'transaction_type': 'wire_transfer',
    })
    result = rc.classify_transaction(tx)

    assert result['scoring_method'] == 'model'
    assert 0.0 <= result['risk_score'] <= 1.0
    assert result['risk_level'] in {'LOW', 'MEDIUM', 'HIGH'}


def test_single_transaction_features_match_training_columns(trained_classifier):
    frame = trained_classifier.build_feature_frame({'amount': 500, 'from_bank': 3, 'to_bank': 3})
    assert list(frame.columns) == trained_classifier.feature_columns
    assert frame['same_bank'].iloc[0] == 1


def test_basic_risk_score_large_amount_branch():
    clf = rc.AdvancedAMLRiskClassifier('__none__.pkl')
    tx = {'amount': 60001, 'timestamp': '2024-01-03T12:00:00'}  # Wednesday midday
    assert clf.get_basic_risk_score(tx) == pytest.approx(0.5)


def test_anomaly_prediction_encodes_like_training():
    """Regression: prediction compared str values to int classes, encoding every bank as -1."""
    df = make_transactions(n=500).drop(columns=['Is_Laundering'])
    detector = AnomalyDetector(contamination=0.05)
    detector.train(df)

    fitted = detector._encode_categorical_features(df, fit=False)
    assert (fitted['From_Bank'] >= 0).all()

    result = detector.predict_anomalies(df.head(1))
    assert result['risk_level'].iloc[0] in {'High Risk', 'Low Risk'}


def test_anomaly_model_loads_old_int_encoders(anomaly_model_dir):
    detector = AnomalyDetector()
    detector.load_model(anomaly_model_dir)
    encoder = detector.encoders['From_Bank']
    encoder.classes_ = np.array(sorted(int(c) for c in encoder.classes_))  # as older models saved them

    encoded = detector._encode_categorical_features(make_transactions(n=50), fit=False)
    assert (encoded['From_Bank'] >= 0).all()
