import pandas as pd

from src.utils.config_loader import config_path

STANDARD_COLUMNS = [
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


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw IBM AML columns ('From Bank', 'Account', ...) to the project's names."""
    df = df.iloc[:, :len(STANDARD_COLUMNS)].copy()
    df.columns = STANDARD_COLUMNS[:df.shape[1]]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


def load_and_clean_data(path: str = None):
    raw_path = path or config_path("raw_data")
    return standardize_columns(pd.read_csv(raw_path))
