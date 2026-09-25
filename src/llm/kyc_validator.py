import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.preprocess_kyc import process_kyc_docs
from src.llm.llm_client import get_llm_client
from src.utils.config_loader import load_config


def build_prompt(record: dict) -> str:
    return f"""
You are a compliance officer reviewing a customer's KYC data.

Please assess the following information:
- Completeness: Are all required fields filled?
- Consistency: Do the fields match each other? (any contradictions?)
- Risk indicators: Are there any red flags or unusual patterns? (e.g., tax haven, fake occupation, illogical source of funds)

KYC Details:
Customer Name: {record['Customer Name']}
Date of Birth: {record['Date of Birth']}
Nationality: {record['Nationality']}
Current Address: {record['Current Address']}
Account Opening Date: {record['Account Opening Date']}
Source of Funds: {record['Source of Funds']}
Occupation: {record['Occupation']}
Red Flags: {record['Red Flags']}

Return a summary report with findings and a risk assessment (low/medium/high).
"""


def validate_kyc(record: dict) -> str:
    prompt = build_prompt(record)
    response = get_llm_client().chat.completions.create(
        model=load_config()["llm"]["model"],
        messages=[
            {"role": "system", "content": "You are an AML compliance assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    kyc_data = process_kyc_docs()
    print(f"Processed {len(kyc_data)} KYC documents.")

    for i, record in enumerate(kyc_data):
        print(f"\n--- Validating KYC #{i+1} ({record['Customer Name']}) ---")
        summary = validate_kyc(record)
        print(summary)
