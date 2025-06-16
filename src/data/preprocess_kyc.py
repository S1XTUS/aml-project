import os
import re

def extract_fields(text: str) -> dict:
    fields = {
        "Customer Name": "",
        "Date of Birth": "",
        "Nationality": "",
        "Current Address": "",
        "Account Opening Date": "",
        "Source of Funds": "",
        "Occupation": "",
        "Red Flags": ""
    }

    for key in fields:
        match = re.search(f"{key}: (.+)", text)
        if match:
            fields[key] = match.group(1).strip()

    return fields

def process_kyc_docs(folder_path="data/kyc_docs/") -> list:
    records = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                content = f.read()
                fields = extract_fields(content)
                fields["filename"] = fname
                records.append(fields)
    return records

if __name__ == "__main__":
    kyc_data = process_kyc_docs()
    print(f"Processed {len(kyc_data)} KYC documents.")
    print(kyc_data[0])
