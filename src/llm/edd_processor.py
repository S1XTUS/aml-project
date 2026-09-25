import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.data.preprocess_kyc import process_kyc_docs
from src.llm.llm_client import get_llm_client
from src.utils.config_loader import load_config

def build_edd_prompt(record: dict) -> str:
    return f"""
You are performing Enhanced Due Diligence (EDD) on a high-risk client.

Using the KYC data below, generate:
1. A summary of risk factors
2. Any specific red flags or typologies observed
3. A risk rating (low / medium / high)
4. Recommended actions

KYC Profile:
Customer Name: {record['Customer Name']}
Date of Birth: {record['Date of Birth']}
Nationality: {record['Nationality']}
Address: {record['Current Address']}
Source of Funds: {record['Source of Funds']}
Occupation: {record['Occupation']}
Account Opening Date: {record['Account Opening Date']}
Red Flags: {record['Red Flags']}
"""

def generate_edd_summary(record: dict) -> str:
    prompt = build_edd_prompt(record)

    response = get_llm_client().chat.completions.create(
        model=load_config()["llm"]["model"],
        messages=[
            {"role": "system", "content": "You are a financial risk expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    kyc_records = process_kyc_docs()

    for i, record in enumerate(kyc_records):
        print(f"\n--- EDD Summary for {record['Customer Name']} ---")
        summary = generate_edd_summary(record)
        print(summary)

        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f"edd_summary_{record['Customer Name'].replace(' ', '_')}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(summary)