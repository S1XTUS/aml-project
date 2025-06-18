import streamlit as st
import requests

st.title("AML Risk Scoring & SAR Generator")

# Shared input fields
case_id = st.text_input("Case ID", "CASE-2025-001")
timestamp = st.text_input("Transaction Time (ISO8601)", "2025-06-18T12:00:00Z")
from_bank = st.text_input("From Bank", "Alpha Bank")
from_account = st.text_input("From Account", "1234567890")
to_bank = st.text_input("To Bank", "Beta Bank")
to_account = st.text_input("To Account", "9876543210")
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
currency = st.text_input("Currency", "USD")

# Risk Score API Call
if st.button("Get Risk Score"):
    risk_payload = {
        "case_id": case_id,
        "transaction_time": timestamp,
        "from_bank": from_bank,
        "from_account": from_account,
        "to_bank": to_bank,
        "to_account": to_account,
        "amount": amount,
        "currency": currency
    }

    response = requests.post("http://127.0.0.1:8000/predict-risk", json=risk_payload)
    if response.status_code == 200:
        risk_score = response.json().get("risk_score")
        st.success(f"Predicted Risk Score: {risk_score:.4f}")
        st.session_state["risk_score"] = risk_score
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")

# SAR Generator
if st.button("Generate SAR"):
    risk_score = st.session_state.get("risk_score", 0.85)

    sar_payload = {
        "case_id": case_id,
        "transaction_time": timestamp,
        "from_bank": from_bank,
        "from_account": from_account,
        "to_bank": to_bank,
        "to_account": to_account,
        "amount": amount,
        "currency": currency,
        "risk_score": risk_score,
        "anomaly_flag": True,
        "pattern_summary": "Large transaction to offshore jurisdiction below reporting threshold.",
        "kyc_summary": "Recent address change to high-risk country. No prior large USD transfers.",
        "regulatory_reference": "FinCEN Advisory FIN-2023-A002"
    }

    response = requests.post("http://127.0.0.1:8000/generate-sar", json=sar_payload)
    if response.status_code == 200:
        sar_narrative = response.json().get("sar_narrative")
        st.text_area("Generated SAR Narrative", sar_narrative, height=300)
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
