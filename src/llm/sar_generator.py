import json
import os
from openai import OpenAI
from typing import Dict

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here"),
    base_url="https://api.deepseek.com/v1"
)

def format_prompt(case: Dict) -> str:
    return f"""
You are a financial crime investigator tasked with writing a Suspicious Activity Report (SAR).

Given the following case data, write a professional, detailed narrative describing why this transaction appears suspicious.

Case Details:
- Case ID: {case['case_id']}
- From Account: {case['from_account']} at {case['from_bank']}
- To Account: {case['to_account']} at {case['to_bank']}
- Amount: {case['amount']} {case['currency']}
- Date & Time: {case['transaction_time']}
- Risk Score: {case['risk_score']}
- Anomaly Detected: {"Yes" if case['anomaly_flag'] else "No"}
- Pattern Summary: {case['pattern_summary']}
- KYC Summary: {case['kyc_summary']}
- Regulatory Reference: {case['regulatory_reference']}

Write this SAR as a formal narrative paragraph.
"""

def generate_sar(case: Dict) -> str:
    prompt = format_prompt(case)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a compliance analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    with open("demo/sample_alert.json") as f:
        case = json.load(f)

    sar_text = generate_sar(case)

    os.makedirs("demo", exist_ok=True)
    with open("demo/sample_sar_output.txt", "w") as f:
        f.write(sar_text)

    print("SAR narrative written to demo/sample_sar_output.txt")
