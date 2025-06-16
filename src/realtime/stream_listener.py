import pandas as pd
import time
import csv
import os
from src.realtime.realtime_scorer import score_transaction

def transaction_stream(file_path, delay=0.5, chunksize=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            yield row
            time.sleep(delay)

output_file = "data/processed/suspicious_transactions.csv"
if not os.path.isfile(output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 
            'Amount Received', 'Receiving Currency', 'Amount Paid', 
            'Payment Currency', 'Payment Format', 'Is Laundering', 
            'Risk Score'
        ])

try:
    for tx in transaction_stream("data/raw/transactions.csv", delay=1.0):
        print("📥 Incoming Transaction:", tx)
        result = score_transaction(tx)
        if result["is_suspicious"]:
            print("🚨 ALERT: Suspicious Transaction Detected!")
            print("Risk Score:", round(result["score"], 3))
            with open(output_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    tx['Timestamp'], tx['From Bank'], tx['Account'], tx['To Bank'], tx['Account.1'],
                    tx['Amount Received'], tx['Receiving Currency'], tx['Amount Paid'], 
                    tx['Payment Currency'], tx['Payment Format'], tx['Is Laundering'],
                    round(result["score"], 3)
                ])
        else:
            print("✅ Transaction passed risk checks.\n")
except KeyboardInterrupt:
    print("\nStreaming stopped by user.")
