from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
from src.models.risk_classifier import predict_risk_score
from src.llm.sar_generator import generate_sar
from src.llm.kyc_validator import validate_kyc
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AML Intelligent Assistant API",
    description="Advanced AML Risk Scoring and SAR Generation API",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to AML Intelligent Assistant API"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AML API"}

class RiskPredictionRequest(BaseModel):
    case_id: str
    transaction_time: str
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount: float
    currency: str
    # Optional fields for enhanced risk assessment
    transaction_type: Optional[str] = "wire_transfer"
    jurisdiction: Optional[str] = "domestic"

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
    # Optional enhanced fields
    transaction_type: Optional[str] = "wire_transfer"
    jurisdiction: Optional[str] = "domestic"

@app.post("/predict-risk")
def predict_risk(request: RiskPredictionRequest):
    """
    Predict the risk score for a given transaction
    
    Returns a risk score between 0 and 1, where:
    - 0.0-0.4: Low risk
    - 0.4-0.7: Medium risk  
    - 0.7-1.0: High risk
    """
    try:
        logger.info(f"Processing risk prediction for case: {request.case_id}")
        
        # Convert the request to the format expected by predict_risk_score
        transaction_data = {
            'case_id': request.case_id,
            'timestamp': request.transaction_time,
            'from_bank': request.from_bank,
            'from_account': request.from_account,
            'to_bank': request.to_bank,
            'to_account': request.to_account,
            'amount': request.amount,
            'amount_paid': request.amount,  # Add both for compatibility
            'amount_received': request.amount,
            'currency': request.currency,
            'receiving_currency': request.currency,
            'payment_currency': request.currency,
            'payment_format': request.transaction_type,
            'transaction_type': request.transaction_type,
            'jurisdiction': request.jurisdiction
        }
        
        # Get risk score from the classifier
        risk_score = predict_risk_score(transaction_data)
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        logger.info(f"Risk prediction completed for case {request.case_id}: {risk_score:.4f} ({risk_level})")
        
        return {
            "case_id": request.case_id,
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "timestamp": request.transaction_time,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error predicting risk for case {request.case_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Risk prediction failed: {str(e)}"
        )

@app.post("/generate-sar")
def generate_sar_endpoint(tx: Transaction):
    """
    Generate a Suspicious Activity Report (SAR) narrative for a transaction
    """
    try:
        logger.info(f"Generating SAR for case: {tx.case_id}")
        
        # Convert Pydantic model to dict
        case_data = tx.dict()
        
        # Generate SAR narrative
        sar_text = generate_sar(case_data)
        
        logger.info(f"SAR generation completed for case {tx.case_id}")
        
        return {
            "case_id": tx.case_id,
            "sar_narrative": sar_text,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error generating SAR for case {tx.case_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"SAR generation failed: {str(e)}"
        )

@app.post("/validate-kyc")
async def validate_kyc_api(file: UploadFile = File(...)):
    """
    Validate KYC documentation
    """
    try:
        logger.info(f"Processing KYC validation for file: {file.filename}")
        
        # Read and decode file content
        content = await file.read()
        text = content.decode("utf-8")
        
        # Validate KYC
        result = validate_kyc(text)
        
        logger.info(f"KYC validation completed for file: {file.filename}")
        
        return {
            "filename": file.filename,
            "validation_result": result,
            "status": "success"
        }
        
    except UnicodeDecodeError:
        logger.error(f"Unable to decode file {file.filename} as UTF-8")
        raise HTTPException(
            status_code=400, 
            detail="File must be a valid text file (UTF-8 encoded)"
        )
    except Exception as e:
        logger.error(f"Error validating KYC for file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"KYC validation failed: {str(e)}"
        )

@app.post("/batch-risk-predict")
def batch_predict_risk(transactions: list[RiskPredictionRequest]):
    """
    Predict risk scores for multiple transactions in batch
    """
    try:
        logger.info(f"Processing batch risk prediction for {len(transactions)} transactions")
        
        results = []
        
        for transaction in transactions:
            try:
                # Convert to transaction data format
                transaction_data = {
                    'case_id': transaction.case_id,
                    'timestamp': transaction.transaction_time,
                    'from_bank': transaction.from_bank,
                    'from_account': transaction.from_account,
                    'to_bank': transaction.to_bank,
                    'to_account': transaction.to_account,
                    'amount': transaction.amount,
                    'amount_paid': transaction.amount,
                    'amount_received': transaction.amount,
                    'currency': transaction.currency,
                    'receiving_currency': transaction.currency,
                    'payment_currency': transaction.currency,
                    'payment_format': transaction.transaction_type,
                    'transaction_type': transaction.transaction_type,
                    'jurisdiction': transaction.jurisdiction
                }
                
                # Get risk score
                risk_score = predict_risk_score(transaction_data)
                
                # Determine risk level
                if risk_score >= 0.7:
                    risk_level = "HIGH"
                elif risk_score >= 0.4:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
                
                results.append({
                    "case_id": transaction.case_id,
                    "risk_score": float(risk_score),
                    "risk_level": risk_level,
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Error processing transaction {transaction.case_id}: {str(e)}")
                results.append({
                    "case_id": transaction.case_id,
                    "risk_score": None,
                    "risk_level": "ERROR",
                    "error": str(e),
                    "status": "failed"
                })
        
        successful_predictions = len([r for r in results if r["status"] == "success"])
        logger.info(f"Batch prediction completed: {successful_predictions}/{len(transactions)} successful")
        
        return {
            "total_transactions": len(transactions),
            "successful_predictions": successful_predictions,
            "results": results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error in batch risk prediction: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model-info")
def get_model_info():
    """
    Get information about the loaded risk prediction model
    """
    try:
        from src.models.risk_classifier import get_classifier
        
        classifier = get_classifier()
        
        model_info = {
            "model_loaded": classifier.model is not None,
            "feature_count": len(classifier.feature_columns) if classifier.feature_columns else 0,
            "optimal_threshold": getattr(classifier, 'optimal_threshold', 0.5),
            "model_type": "XGBoost" if classifier.model is not None else "Rule-based",
            "status": "ready"
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get model info: {str(e)}"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=True  # Enable auto-reload during development
    )