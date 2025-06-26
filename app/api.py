from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
from src.models.risk_classifier import predict_risk_score
from src.llm.sar_generator import IntegratedSARGenerator  # Updated import
from src.llm.kyc_validator import validate_kyc
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AML Intelligent Assistant API",
    description="Advanced AML Risk Scoring and SAR Generation API with Integrated Analysis",
    version="2.0.0"
)

# Initialize the integrated SAR generator
sar_generator = IntegratedSARGenerator()

@app.get("/")
def read_root():
    return {"message": "Welcome to AML Intelligent Assistant API v2.0 - Integrated Analysis"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AML API v2.0", "features": ["risk_prediction", "anomaly_detection", "model_explanation", "integrated_sar"]}

class RiskPredictionRequest(BaseModel):
    case_id: str
    transaction_time: str
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount: float
    currency: str
    # Enhanced fields for comprehensive analysis
    transaction_type: Optional[str] = "wire_transfer"
    jurisdiction: Optional[str] = "domestic"
    recipient_country: Optional[str] = "domestic"
    transaction_frequency: Optional[int] = 1
    kyc_completeness: Optional[float] = 1.0
    account_age: Optional[int] = 365
    beneficiary_type: Optional[str] = "individual"

class EnhancedTransaction(BaseModel):
    case_id: str
    from_account: str
    from_bank: str
    to_account: str
    to_bank: str
    amount: float
    currency: str
    transaction_time: str
    # Enhanced fields for comprehensive SAR generation
    transaction_type: Optional[str] = "wire_transfer"
    jurisdiction: Optional[str] = "domestic"
    recipient_country: Optional[str] = "domestic"
    transaction_frequency: Optional[int] = 1
    kyc_completeness: Optional[float] = 1.0
    account_age: Optional[int] = 365
    beneficiary_type: Optional[str] = "individual"
    pattern_summary: Optional[str] = "Standard transaction pattern"
    kyc_summary: Optional[str] = "KYC documentation complete"
    customer_notes: Optional[str] = "No additional notes"
    regulatory_reference: Optional[str] = "BSA/AML compliance requirements"

class ComprehensiveAnalysisRequest(BaseModel):
    """Request for full transaction analysis including risk, anomaly, and explanation"""
    case_id: str
    transaction_time: str
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount: float
    currency: str
    transaction_type: Optional[str] = "wire_transfer"
    jurisdiction: Optional[str] = "domestic"
    recipient_country: Optional[str] = "domestic"
    transaction_frequency: Optional[int] = 1
    kyc_completeness: Optional[float] = 1.0
    account_age: Optional[int] = 365
    beneficiary_type: Optional[str] = "individual"
    pattern_summary: Optional[str] = "Standard transaction pattern"
    kyc_summary: Optional[str] = "KYC documentation complete"
    customer_notes: Optional[str] = "No additional notes"

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
            'amount_paid': request.amount,
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

@app.post("/analyze-comprehensive")
def analyze_comprehensive(request: ComprehensiveAnalysisRequest):
    """
    Perform comprehensive analysis including risk assessment, anomaly detection, and model explanation
    """
    try:
        logger.info(f"Processing comprehensive analysis for case: {request.case_id}")
        
        # Convert request to transaction data format
        transaction_data = request.dict()
        
        # Run comprehensive analysis using integrated system
        analysis_results = sar_generator.analyze_transaction(transaction_data)
        
        # Extract results from each component
        explanation_result = analysis_results['explanation']
        anomaly_result = analysis_results['anomaly']
        risk_result = analysis_results['risk']
        
        # Format response
        response = {
            "case_id": request.case_id,
            "timestamp": datetime.now().isoformat(),
            "risk_assessment": {
                "risk_level": risk_result.risk_level,
                "risk_score": risk_result.risk_score,
                "risk_factors": risk_result.risk_factors,
                "regulatory_flags": risk_result.regulatory_flags,
                "recommendation": risk_result.recommendation
            },
            "anomaly_detection": {
                "anomaly_detected": anomaly_result.anomaly_flag,
                "anomaly_score": anomaly_result.anomaly_score,
                "anomaly_type": anomaly_result.anomaly_type,
                "detected_patterns": anomaly_result.detected_patterns,
                "baseline_deviation": anomaly_result.baseline_deviation
            },
            "model_explanation": {
                "top_features": dict(explanation_result.top_features[:5]),
                "confidence_score": explanation_result.confidence_score,
                "explanation_text": explanation_result.explanation_text
            },
            "status": "success"
        }
        
        logger.info(f"Comprehensive analysis completed for case {request.case_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis for case {request.case_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Comprehensive analysis failed: {str(e)}"
        )

@app.post("/generate-sar")
def generate_sar_endpoint(tx: EnhancedTransaction):
    """
    Generate a comprehensive Suspicious Activity Report (SAR) narrative using integrated analysis
    """
    try:
        logger.info(f"Generating comprehensive SAR for case: {tx.case_id}")
        
        # Convert Pydantic model to dict
        case_data = tx.dict()
        
        # Generate comprehensive SAR using integrated system
        sar_output = sar_generator.generate_enhanced_sar(case_data)
        
        # Extract analysis results for API response
        analysis_results = sar_output['analysis_results']
        
        response = {
            "case_id": tx.case_id,
            "sar_narrative": sar_output['sar_narrative'],
            "generation_timestamp": sar_output['generation_timestamp'],
            "analysis_summary": {
                "risk_level": analysis_results['risk'].risk_level,
                "risk_score": analysis_results['risk'].risk_score,
                "anomaly_detected": analysis_results['anomaly'].anomaly_flag,
                "anomaly_score": analysis_results['anomaly'].anomaly_score,
                "model_confidence": analysis_results['explanation'].confidence_score,
                "top_risk_factors": analysis_results['risk'].risk_factors[:3],
                "detected_patterns": analysis_results['anomaly'].detected_patterns,
                "regulatory_flags": analysis_results['risk'].regulatory_flags
            },
            "status": "success"
        }
        
        logger.info(f"Comprehensive SAR generation completed for case {tx.case_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating comprehensive SAR for case {tx.case_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"SAR generation failed: {str(e)}"
        )

@app.post("/generate-sar-full")
def generate_sar_full_output(tx: EnhancedTransaction):
    """
    Generate complete SAR output including all analysis details and save files
    """
    try:
        logger.info(f"Generating full SAR output for case: {tx.case_id}")
        
        # Convert Pydantic model to dict
        case_data = tx.dict()
        
        # Generate comprehensive SAR with file outputs
        sar_output = sar_generator.generate_enhanced_sar(case_data)
        
        # Save comprehensive output files
        sar_generator.save_comprehensive_output(sar_output, output_dir="api_outputs")
        
        # Return complete analysis
        response = {
            "case_id": tx.case_id,
            "sar_narrative": sar_output['sar_narrative'],
            "generation_timestamp": sar_output['generation_timestamp'],
            "comprehensive_analysis": {
                "risk_assessment": {
                    "risk_level": sar_output['analysis_results']['risk'].risk_level,
                    "risk_score": sar_output['analysis_results']['risk'].risk_score,
                    "risk_factors": sar_output['analysis_results']['risk'].risk_factors,
                    "regulatory_flags": sar_output['analysis_results']['risk'].regulatory_flags,
                    "recommendation": sar_output['analysis_results']['risk'].recommendation,
                    "compliance_notes": sar_output['analysis_results']['risk'].compliance_notes
                },
                "anomaly_detection": {
                    "anomaly_detected": sar_output['analysis_results']['anomaly'].anomaly_flag,
                    "anomaly_score": sar_output['analysis_results']['anomaly'].anomaly_score,
                    "anomaly_type": sar_output['analysis_results']['anomaly'].anomaly_type,
                    "detected_patterns": sar_output['analysis_results']['anomaly'].detected_patterns,
                    "baseline_deviation": sar_output['analysis_results']['anomaly'].baseline_deviation,
                    "time_series_anomalies": sar_output['analysis_results']['anomaly'].time_series_anomalies
                },
                "model_explanation": {
                    "top_features": dict(sar_output['analysis_results']['explanation'].top_features),
                    "feature_importance_scores": sar_output['analysis_results']['explanation'].feature_importance_scores,
                    "confidence_score": sar_output['analysis_results']['explanation'].confidence_score,
                    "explanation_text": sar_output['analysis_results']['explanation'].explanation_text
                }
            },
            "files_generated": [
                f"api_outputs/sar_narrative_{tx.case_id}.txt",
                f"api_outputs/analysis_summary_{tx.case_id}.json",
                f"api_outputs/complete_output_{tx.case_id}.json"
            ],
            "status": "success"
        }
        
        logger.info(f"Full SAR output generation completed for case {tx.case_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating full SAR output for case {tx.case_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Full SAR generation failed: {str(e)}"
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

@app.post("/batch-comprehensive-analysis")
def batch_comprehensive_analysis(transactions: list[ComprehensiveAnalysisRequest]):
    """
    Perform comprehensive analysis for multiple transactions in batch
    """
    try:
        logger.info(f"Processing batch comprehensive analysis for {len(transactions)} transactions")
        
        results = []
        
        for transaction in transactions:
            try:
                # Convert to transaction data format
                transaction_data = transaction.dict()
                
                # Run comprehensive analysis
                analysis_results = sar_generator.analyze_transaction(transaction_data)
                
                # Extract results
                explanation_result = analysis_results['explanation']
                anomaly_result = analysis_results['anomaly']
                risk_result = analysis_results['risk']
                
                results.append({
                    "case_id": transaction.case_id,
                    "risk_level": risk_result.risk_level,
                    "risk_score": risk_result.risk_score,
                    "anomaly_detected": anomaly_result.anomaly_flag,
                    "anomaly_score": anomaly_result.anomaly_score,
                    "model_confidence": explanation_result.confidence_score,
                    "top_risk_factors": risk_result.risk_factors[:3],
                    "regulatory_flags": risk_result.regulatory_flags,
                    "recommendation": risk_result.recommendation,
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Error processing transaction {transaction.case_id}: {str(e)}")
                results.append({
                    "case_id": transaction.case_id,
                    "error": str(e),
                    "status": "failed"
                })
        
        successful_analyses = len([r for r in results if r["status"] == "success"])
        logger.info(f"Batch comprehensive analysis completed: {successful_analyses}/{len(transactions)} successful")
        
        return {
            "total_transactions": len(transactions),
            "successful_analyses": successful_analyses,
            "results": results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error in batch comprehensive analysis: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Batch comprehensive analysis failed: {str(e)}"
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
    Get information about the loaded models and integrated system
    """
    try:
        from src.models.risk_classifier import get_classifier
        
        classifier = get_classifier()
        
        model_info = {
            "risk_classifier": {
                "model_loaded": classifier.model is not None,
                "feature_count": len(classifier.feature_columns) if classifier.feature_columns else 0,
                "optimal_threshold": getattr(classifier, 'optimal_threshold', 0.5),
                "model_type": "XGBoost" if classifier.model is not None else "Rule-based"
            },
            "integrated_system": {
                "explainer_component": "Active",
                "anomaly_detector": "Active", 
                "risk_classifier": "Active",
                "sar_generator": "Active"
            },
            "api_version": "2.0.0",
            "features": [
                "risk_prediction",
                "anomaly_detection", 
                "model_explanation",
                "integrated_sar_generation",
                "comprehensive_analysis",
                "batch_processing"
            ],
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