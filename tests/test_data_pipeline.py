import pandas as pd

from src.data.load_data import standardize_columns, STANDARD_COLUMNS
from src.data.preprocess_kyc import extract_fields, process_kyc_docs
from src.utils.config_loader import load_config, risk_level_from_score


def test_config_thresholds_are_numeric():
    thresholds = load_config()["thresholds"]
    assert isinstance(thresholds["risk_high"], float)
    assert isinstance(thresholds["risk_medium"], float)
    assert thresholds["risk_medium"] < thresholds["risk_high"]


def test_config_has_no_api_keys():
    assert "api_keys" not in load_config()


def test_risk_level_from_score():
    assert risk_level_from_score(0.95) == "HIGH"
    assert risk_level_from_score(0.5) == "MEDIUM"
    assert risk_level_from_score(0.1) == "LOW"


def test_standardize_columns_renames_raw_ibm_columns():
    raw = pd.DataFrame([["2022/09/01 00:08", 11, "8000ECA90", 11, "8000ECA90", 100.0,
                         "US Dollar", 100.0, "US Dollar", "Reinvestment", 0]],
                       columns=["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
                                "Amount Received", "Receiving Currency", "Amount Paid",
                                "Payment Currency", "Payment Format", "Is Laundering"])
    df = standardize_columns(raw)
    assert list(df.columns) == STANDARD_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"])


def test_extract_fields_reads_kyc_text():
    text = "Customer Name: Jane Doe\nNationality: Panama\nOccupation: Consultant\n"
    fields = extract_fields(text)
    assert fields["Customer Name"] == "Jane Doe"
    assert fields["Nationality"] == "Panama"
    assert fields["Source of Funds"] == ""


def test_process_kyc_docs_finds_sample_docs():
    records = process_kyc_docs()
    assert records
    assert all("filename" in r for r in records)
