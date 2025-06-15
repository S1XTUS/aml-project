import os
from openai import OpenAI
from src.data.preprocess_kyc import process_kyc_docs

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url= "https://api.deepseek.com/v1"
)

def build_prommt(record:dict) -> str:
    prompt = f"""
You are a compliance officer reviewing a customer's KYC data.set

Please assess the following information:
- Completeness: Are all required fields filled?
- Consistency: Do the fields match each other? (any contradictions?)
- Risk indicators : Are there any red flags or unusual patterns? ((e.g., tax haven, fake occupation, illogical source of funds)

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
    prompt = build_prommt(record)
    resposne = client.chat.completions.create(
        model="deepseek-chat",
       messages=[
            {"role": "system", "content": "You are an AML compliance assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return resposne.choices[0].message.content


if __name__ == "__main__":
    kyc_data = process_kyc_docs()
    print(f"Processed {len(kyc_data)} KYC documents.")
    
    for i , record in enumerate(kyc_data):
        print(f"\n--- Validating KYC #{i+1} ({record['Customer Name']}) ---")
        summary = validate_kyc(record)
        print(summary)