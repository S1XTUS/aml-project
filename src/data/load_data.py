import pandas as pd
import yaml
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
config_path = os.path.join(project_root, "config.yaml")


with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def load_and_clean_data():
    raw_path = os.path.join(project_root, config['paths']['raw_data'])
    df = pd.read_csv(raw_path)

    df.columns = [
        "Timestamp",
        "From_Bank",
        "From_Account",
        "To_Bank",
        "To_Account",
        "Amount_Received",
        "Receiving_Currency",
        "Amount_Paid",
        "Payment_Currency",
        "Payment_Format",
        "Is_Laundering"
    ]

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    return df
