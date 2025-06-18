from fastapi import FastAPI, UploadFile, File , HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
# from src.models.risk_classifier import predict_risk_score
from src.llm.sar_generator import generate_sar
from src.llm.kyc_validator import validate_kyc
from typing import Optional

app = FastAPI(title="AML Intelligent Assistant API")

@app.get("/")
def read_root():
    return {"message": "Welcome to AML Intelligent Assistant API"}


class Transaction(BaseModel):
    case_id: str
    from_account: str
    from_bank: str
    to_account: str
    to_bank: str
    amount: float
    currency: str
    transaction_time: str
    risk_score: float
    anomaly_flag: bool
    pattern_summary: str
    kyc_summary: str
    regulatory_reference: str
    
# @app.post("/predict-risk")
# def predict_risk(tx: Transaction):
#     # Convert to DataFrame
#     df = pd.DataFrame([tx.dict()])
#     score = predict_risk_score(df)
#     return {"risk_score": float(score)}

@app.post("/generate-sar")
def generate_sar_endpoint(tx: Transaction):
    try:
        # Convert Pydantic model to dict if needed
        case_data = tx.dict()

        sar_text = generate_sar(case_data)  # your existing SAR generation function
        return {"sar_narrative": sar_text}
    except Exception as e:
        print(f"Error generating SAR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate-kyc")
async def validate_kyc_api(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    result = validate_kyc(text)
    return {"validation_result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    