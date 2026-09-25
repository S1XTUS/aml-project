"""
Build a small file of real, labelled transactions for testing the dashboard / API.

Sources (all synthetic AML benchmarks):
  ibm_li_small_test  rows from the risk classifier's held-out test split (in-distribution)
  ibm_hi_small       IBM HI-Small: same simulator, different world, higher laundering rate
  saml_d             SAML-D (Kaggle berkanoztas/synthetic-transaction-monitoring-dataset-aml)
  amlsim             IBM AMLSim example (Kaggle anshankul/ibm-amlsim-example-dataset)

Every row is mapped to the dashboard's fields (bank, account, amount, currency code,
transaction type, jurisdiction) and keeps its ground-truth label and typology.

Usage:  python src/data/build_test_samples.py
Output: data/samples/test_transactions.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.load_data import standardize_columns
from src.utils.config_loader import config_path, project_path

OUTPUT_PATH = project_path("data", "samples", "test_transactions.csv")
LAUNDERING_PER_SOURCE = 300
NORMAL_PER_SOURCE = 700
SEED = 42

# Dataset currency names -> ISO codes used by the dashboard
CURRENCY_CODES = {
    # IBM
    'US Dollar': 'USD', 'Euro': 'EUR', 'UK Pound': 'GBP', 'Swiss Franc': 'CHF', 'Yen': 'JPY',
    'Canadian Dollar': 'CAD', 'Australian Dollar': 'AUD', 'Yuan': 'CNY', 'Rupee': 'INR',
    'Ruble': 'RUB', 'Brazil Real': 'BRL', 'Mexican Peso': 'MXN', 'Saudi Riyal': 'SAR',
    'Shekel': 'ILS', 'Bitcoin': 'BTC',
    # SAML-D
    'UK pounds': 'GBP', 'US dollar': 'USD', 'Swiss franc': 'CHF', 'Indian rupee': 'INR',
    'Dirham': 'AED', 'Moroccan dirham': 'MAD', 'Naira': 'NGN', 'Pakistani rupee': 'PKR',
    'Turkish lira': 'TRY', 'Albanian lek': 'ALL',
}

# Dataset payment formats -> dashboard transaction type values
PAYMENT_TYPES = {
    # IBM
    'Wire': 'wire_transfer', 'ACH': 'ach', 'Cheque': 'check', 'Cash': 'cash',
    'Credit Card': 'credit_card', 'Bitcoin': 'bitcoin', 'Reinvestment': 'reinvestment',
    # SAML-D
    'Cash Deposit': 'cash_deposit', 'Cash Withdrawal': 'cash_withdrawal',
    'Credit card': 'credit_card', 'Debit card': 'debit_card', 'Cross-border': 'international_transfer',
    # AMLSim
    'TRANSFER': 'wire_transfer',
}

EU_COUNTRIES = {'Austria', 'France', 'Germany', 'Italy', 'Netherlands', 'Spain'}

OUTPUT_COLUMNS = ['source', 'record_id', 'timestamp', 'from_bank', 'from_account', 'to_bank',
                  'to_account', 'amount', 'currency', 'transaction_type', 'jurisdiction',
                  'receiver_country', 'is_laundering', 'typology']


def _sample(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    positives = df[df[label_col] == 1]
    negatives = df[df[label_col] == 0]
    return pd.concat([
        positives.sample(min(LAUNDERING_PER_SOURCE, len(positives)), random_state=SEED),
        negatives.sample(min(NORMAL_PER_SOURCE, len(negatives)), random_state=SEED),
    ])


def _ibm_typologies(patterns_path: str) -> dict:
    """Map each laundering transaction in an IBM *_Patterns.txt file to its typology."""
    typologies = {}
    current = None
    with open(patterns_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('BEGIN LAUNDERING ATTEMPT'):
                current = re.sub(r'^BEGIN LAUNDERING ATTEMPT - ', '', line).split(':')[0].strip()
            elif line.startswith('END LAUNDERING ATTEMPT'):
                current = None
            elif current and line:
                parts = line.split(',')
                key = (pd.Timestamp(parts[0].replace('/', '-')), int(parts[1]), parts[2],
                       int(parts[3]), parts[4], round(float(parts[7]), 2))
                typologies[key] = current
    return typologies


def _from_ibm(df: pd.DataFrame, source: str, patterns_path: str) -> pd.DataFrame:
    typologies = _ibm_typologies(patterns_path)
    keys = zip(df['Timestamp'], df['From_Bank'], df['From_Account'], df['To_Bank'],
               df['To_Account'], df['Amount_Paid'].round(2))
    out = pd.DataFrame({
        'source': source,
        'record_id': df.index.astype(str),
        'timestamp': df['Timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S'),
        'from_bank': df['From_Bank'].astype(str),
        'from_account': df['From_Account'],
        'to_bank': df['To_Bank'].astype(str),
        'to_account': df['To_Account'],
        'amount': df['Amount_Paid'].round(2),
        'currency': df['Payment_Currency'].map(CURRENCY_CODES),
        'transaction_type': df['Payment_Format'].map(PAYMENT_TYPES),
        'jurisdiction': 'Domestic',
        'receiver_country': '',
        'is_laundering': df['Is_Laundering'].astype(int),
    })
    out['typology'] = [typologies.get(k, 'UNLISTED PATTERN' if lab else 'NORMAL')
                       for k, lab in zip(keys, out['is_laundering'])]
    return out


def ibm_li_small_test() -> pd.DataFrame:
    """Rows the risk classifier never trained on (same split as train_model)."""
    df = standardize_columns(pd.read_csv(config_path("raw_data")))
    y = df['Is_Laundering'].astype(int).values
    _, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
    sample = _sample(df.iloc[test_idx], 'Is_Laundering')
    return _from_ibm(sample, 'ibm_li_small_test',
                     config_path("raw_data").replace('_Trans.csv', '_Patterns.txt'))


def ibm_hi_small() -> pd.DataFrame:
    path = project_path("data", "raw", "archive", "HI-Small_Trans.csv")
    df = standardize_columns(pd.read_csv(path))
    return _from_ibm(_sample(df, 'Is_Laundering'), 'ibm_hi_small', path.replace('_Trans.csv', '_Patterns.txt'))


def saml_d() -> pd.DataFrame:
    df = pd.read_csv(project_path("data", "external", "saml_d", "SAML-D.csv"))
    df = _sample(df, 'Is_laundering')
    receiver = df['Receiver_bank_location']
    return pd.DataFrame({
        'source': 'saml_d',
        'record_id': df.index.astype(str),
        'timestamp': pd.to_datetime(df['Date'] + ' ' + df['Time']).dt.strftime('%Y-%m-%dT%H:%M:%S'),
        'from_bank': df['Sender_bank_location'] + ' bank',
        'from_account': df['Sender_account'].astype(str),
        'to_bank': receiver + ' bank',
        'to_account': df['Receiver_account'].astype(str),
        'amount': df['Amount'].round(2),
        'currency': df['Payment_currency'].map(CURRENCY_CODES),
        'transaction_type': df['Payment_type'].map(PAYMENT_TYPES),
        'jurisdiction': np.where(receiver == 'UK', 'Domestic',
                                 np.where(receiver.isin(EU_COUNTRIES), 'EU', 'International')),
        'receiver_country': receiver,
        'is_laundering': df['Is_laundering'].astype(int),
        'typology': df['Laundering_type'],
    })


def amlsim() -> pd.DataFrame:
    folder = project_path("data", "external", "amlsim")
    tx = pd.read_csv(os.path.join(folder, "transactions.csv"))
    alerts = pd.read_csv(os.path.join(folder, "alerts.csv"))[['ALERT_ID', 'ALERT_TYPE']].drop_duplicates('ALERT_ID')
    tx['IS_FRAUD'] = tx['IS_FRAUD'].astype(int)
    tx = _sample(tx, 'IS_FRAUD').merge(alerts, on='ALERT_ID', how='left')
    # AMLSim only records a day step; place each transaction at noon on that day
    timestamps = pd.Timestamp('2022-01-01 12:00') + pd.to_timedelta(tx['TIMESTAMP'], unit='D')
    return pd.DataFrame({
        'source': 'amlsim',
        'record_id': tx['TX_ID'].astype(str),
        'timestamp': timestamps.dt.strftime('%Y-%m-%dT%H:%M:%S'),
        'from_bank': 'AMLSim bank',
        'from_account': tx['SENDER_ACCOUNT_ID'].astype(str),
        'to_bank': 'AMLSim bank',
        'to_account': tx['RECEIVER_ACCOUNT_ID'].astype(str),
        'amount': tx['TX_AMOUNT'].round(2),
        'currency': 'USD',
        'transaction_type': tx['TX_TYPE'].map(PAYMENT_TYPES),
        'jurisdiction': 'Domestic',
        'receiver_country': 'US',
        'is_laundering': tx['IS_FRAUD'],
        'typology': tx['ALERT_TYPE'].fillna('normal').str.upper(),
    })


def main():
    frames = []
    for build in (ibm_li_small_test, ibm_hi_small, saml_d, amlsim):
        try:
            frame = build()
            frames.append(frame)
            print(f"{build.__name__}: {len(frame)} rows ({frame['is_laundering'].sum()} laundering)")
        except FileNotFoundError as e:
            print(f"{build.__name__}: skipped, file not found ({e.filename})")

    samples = pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS]
    unmapped = samples[samples['currency'].isna() | samples['transaction_type'].isna()]
    if not unmapped.empty:
        raise ValueError(f"{len(unmapped)} rows have unmapped currency / transaction type")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    samples.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(samples)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
