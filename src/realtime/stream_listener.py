import os
import sys
import csv
import time

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.realtime.realtime_scorer import score_transaction
from src.utils.config_loader import config_path, project_path

OUTPUT_FILE = project_path("data", "processed", "suspicious_transactions.csv")
OUTPUT_COLUMNS = [
    'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1',
    'Amount Received', 'Receiving Currency', 'Amount Paid',
    'Payment Currency', 'Payment Format', 'Is Laundering'
]


def transaction_stream(file_path, delay=0.5, chunksize=10000):
    """Replay a raw IBM AML transactions CSV as a stream."""
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            yield row
            time.sleep(delay)


def ensure_output_file(output_file=OUTPUT_FILE):
    if not os.path.isfile(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(OUTPUT_COLUMNS + ['Risk Score'])


def run(source_file=None, delay=1.0, output_file=OUTPUT_FILE):
    ensure_output_file(output_file)
    try:
        for tx in transaction_stream(source_file or config_path("raw_data"), delay=delay):
            print("📥 Incoming Transaction:", tx.to_dict())
            result = score_transaction(tx)
            if result["is_suspicious"]:
                print("🚨 ALERT: Suspicious Transaction Detected!")
                print("Risk Score:", round(result["score"], 3))
                with open(output_file, mode='a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([tx[col] for col in OUTPUT_COLUMNS] + [round(result["score"], 3)])
            else:
                print("✅ Transaction passed risk checks.\n")
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")


if __name__ == "__main__":
    run()
