import json
import os
import numpy as np
from openai import OpenAI
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here"),
    base_url="https://api.deepseek.com/v1"
)

@dataclass
class ExplanationResult:
    """Output from explainer.py"""
    top_features: List[Tuple[str, float]]
    feature_importance_scores: Dict[str, float]
    explanation_text: str
    confidence_score: float

@dataclass
class AnomalyResult:
    """Output from anomaly detector"""
    anomaly_flag: bool
    anomaly_score: float
    anomaly_type: str
    detected_patterns: List[str]
    baseline_deviation: float
    time_series_anomalies: List[Dict]

@dataclass
class RiskClassificationResult:
    """Output from risk_classifier.py"""
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score: float
    risk_factors: List[str]
    regulatory_flags: List[str]
    compliance_notes: str
    recommendation: str

class ExplainerComponent:
    """Mock implementation of explainer.py functionality"""
    
    def __init__(self):
        self.feature_weights = {
            'transaction_amount': 0.25,
            'recipient_country_risk': 0.20,
            'account_age': 0.15,
            'transaction_frequency': 0.15,
            'kyc_completeness': 0.10,
            'beneficiary_type': 0.10,
            'time_of_transaction': 0.05
        }
    
    def explain_prediction(self, transaction_data: Dict) -> ExplanationResult:
        """Generate explanation for a transaction's risk assessment"""
        
        # Calculate feature contributions
        feature_scores = {}
        for feature, weight in self.feature_weights.items():
            if feature in transaction_data:
                # Get raw value and convert to numeric
                raw_value = transaction_data.get(feature, 0)
                numeric_value = self._convert_to_numeric(raw_value, feature)
                normalized_score = min(numeric_value * weight, 1.0)
                feature_scores[feature] = normalized_score
        
        # Get top contributing features
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate explanation text
        explanation_parts = []
        for feature, score in top_features[:3]:
            if score > 0.15:
                explanation_parts.append(f"{feature.replace('_', ' ').title()}: {score:.2f}")
        
        explanation_text = f"Primary risk drivers: {', '.join(explanation_parts)}"
        confidence_score = sum(score for _, score in top_features) / len(top_features)
        
        return ExplanationResult(
            top_features=top_features,
            feature_importance_scores=feature_scores,
            explanation_text=explanation_text,
            confidence_score=confidence_score
        )
    
    def _convert_to_numeric(self, value: Any, feature_name: str) -> float:
        """Convert various data types to numeric values for analysis"""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Handle specific string-to-numeric conversions based on feature
            if feature_name == 'recipient_country_risk':
                risk_mapping = {
                    'low': 0.1, 'medium': 0.5, 'high': 0.8, 'critical': 1.0,
                    'offshore': 0.9, 'sanctions': 1.0, 'high-risk': 0.8
                }
                return risk_mapping.get(value.lower(), 0.3)
            
            elif feature_name == 'beneficiary_type':
                type_mapping = {
                    'individual': 0.2, 'corporate': 0.4, 'trust': 0.6,
                    'shell': 0.9, 'pep': 0.8, 'unknown': 0.7
                }
                return type_mapping.get(value.lower(), 0.3)
            
            # Try to extract numeric value from string
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        # Default for any other type
        return 0.0

class AnomalyDetector:
    """Mock implementation of anomaly detector functionality"""
    
    def __init__(self):
        self.threshold = 0.7
        
    def detect_anomalies(self, transaction_data: Dict, historical_data: List[Dict] = None) -> AnomalyResult:
        """Detect anomalies in transaction patterns"""
        
        amount = float(transaction_data.get('amount', 0))
        
        # Simulate anomaly detection logic
        anomaly_score = 0.0
        detected_patterns = []
        
        # Amount-based anomaly
        if amount > 50000:
            anomaly_score += 0.4
            detected_patterns.append("Unusually high transaction amount")
        
        # Time-based anomaly
        transaction_time = transaction_data.get('transaction_time', '')
        if 'T' in transaction_time:
            try:
                hour = int(transaction_time.split('T')[1].split(':')[0])
                if hour < 6 or hour > 22:
                    anomaly_score += 0.3
                    detected_patterns.append("Off-hours transaction timing")
            except (ValueError, IndexError):
                pass
        
        # Geographic anomaly
        recipient_country = str(transaction_data.get('recipient_country', '')).lower()
        if recipient_country in ['offshore', 'high-risk', 'sanctions']:
            anomaly_score += 0.5
            detected_patterns.append("High-risk jurisdiction recipient")
        
        # Pattern frequency anomaly
        frequency = float(transaction_data.get('transaction_frequency', 0))
        if frequency > 10:
            anomaly_score += 0.3
            detected_patterns.append("High frequency transaction pattern")
        
        anomaly_flag = anomaly_score >= self.threshold
        anomaly_type = "CRITICAL" if anomaly_score > 0.8 else "MODERATE" if anomaly_score > 0.5 else "LOW"
        
        # Simulate time series anomalies
        time_series_anomalies = []
        if anomaly_flag:
            time_series_anomalies = [
                {"timestamp": "2024-01-15T14:30:00", "deviation": 2.5},
                {"timestamp": "2024-01-15T14:31:00", "deviation": 3.1}
            ]
        
        return AnomalyResult(
            anomaly_flag=anomaly_flag,
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            detected_patterns=detected_patterns,
            baseline_deviation=anomaly_score * 2.5,
            time_series_anomalies=time_series_anomalies
        )

class RiskClassifier:
    """Mock implementation of risk_classifier.py functionality"""
    
    def __init__(self):
        self.risk_thresholds = {
            'LOW': 0.3,
            'MEDIUM': 0.6,
            'HIGH': 0.8,
            'CRITICAL': 1.0
        }
    
    def classify_risk(self, transaction_data: Dict, anomaly_result: AnomalyResult) -> RiskClassificationResult:
        """Classify transaction risk level"""
        
        base_risk = 0.0
        risk_factors = []
        regulatory_flags = []
        
        # Amount-based risk
        amount = float(transaction_data.get('amount', 0))
        if amount > 100000:
            base_risk += 0.4
            risk_factors.append("Large transaction amount exceeds reporting threshold")
            regulatory_flags.append("CTR_REQUIRED")
        elif amount > 10000:
            base_risk += 0.2
            risk_factors.append("Significant transaction amount")
        
        # Geographic risk
        recipient_country = str(transaction_data.get('recipient_country', '')).lower()
        if recipient_country in ['offshore', 'high-risk', 'sanctions']:
            base_risk += 0.3
            risk_factors.append("High-risk jurisdiction involvement")
            regulatory_flags.append("SANCTIONS_CHECK")
        
        # Customer risk
        kyc_score = float(transaction_data.get('kyc_completeness', 1.0))
        if kyc_score < 0.7:
            base_risk += 0.2
            risk_factors.append("Incomplete KYC documentation")
            regulatory_flags.append("KYC_REVIEW")
        
        # Incorporate anomaly findings
        if anomaly_result.anomaly_flag:
            base_risk += anomaly_result.anomaly_score * 0.5
            risk_factors.extend(anomaly_result.detected_patterns)
        
        # Determine risk level
        risk_score = min(base_risk, 1.0)
        risk_level = "LOW"
        for level, threshold in self.risk_thresholds.items():
            if risk_score <= threshold:
                risk_level = level
                break
        
        # Generate compliance notes and recommendations
        compliance_notes = f"Risk assessment based on {len(risk_factors)} identified factors"
        
        recommendations = {
            'LOW': "Monitor transaction, no immediate action required",
            'MEDIUM': "Enhanced monitoring recommended, review customer profile",
            'HIGH': "Immediate review required, consider filing SAR",
            'CRITICAL': "File SAR immediately, consider account restrictions"
        }
        
        return RiskClassificationResult(
            risk_level=risk_level,
            risk_score=risk_score,
            risk_factors=risk_factors,
            regulatory_flags=regulatory_flags,
            compliance_notes=compliance_notes,
            recommendation=recommendations[risk_level]
        )

class IntegratedSARGenerator:
    """Main class that combines all three components for SAR generation"""
    
    def __init__(self):
        self.explainer = ExplainerComponent()
        self.anomaly_detector = AnomalyDetector()
        self.risk_classifier = RiskClassifier()
    
    def analyze_transaction(self, transaction_data: Dict) -> Dict[str, Any]:
        """Run all three components and combine results"""
        
        logger.info(f"Analyzing transaction {transaction_data.get('case_id', 'Unknown')}")
        
        # Run explainer
        explanation_result = self.explainer.explain_prediction(transaction_data)
        logger.info("Explainer analysis completed")
        
        # Run anomaly detection
        anomaly_result = self.anomaly_detector.detect_anomalies(transaction_data)
        logger.info("Anomaly detection completed")
        
        # Run risk classification
        risk_result = self.risk_classifier.classify_risk(transaction_data, anomaly_result)
        logger.info("Risk classification completed")
        
        return {
            'explanation': explanation_result,
            'anomaly': anomaly_result,
            'risk': risk_result
        }
    
    def format_concise_prompt(self, case: Dict, analysis_results: Dict) -> str:
        """Create concise, analyst-friendly prompt for SAR generation"""
        explanation = analysis_results['explanation']
        anomaly = analysis_results['anomaly']
        risk = analysis_results['risk']
            
        return f"""
    URGENT: Write SAR for analyst review - 30 second read time MAX.

    TRANSACTION DATA:
    • ${case['amount']:,} {case.get('currency', 'USD')} 
    • {case['from_account']} → {case['to_account']}
    • {case['transaction_time'][:16].replace('T', ' ')}
    • Destination: {case.get('recipient_country', 'Domestic')}

    RISK ASSESSMENT: {risk.risk_level} RISK ({risk.risk_score:.2f}/1.0)

    RED FLAGS DETECTED:
    {chr(10).join([f'• {flag}' for flag in risk.risk_factors[:4]])}

    ANOMALIES: {' | '.join(anomaly.detected_patterns[:2]) if anomaly.detected_patterns else 'Standard pattern'}

    REGULATORY: {', '.join(risk.regulatory_flags) if risk.regulatory_flags else 'No immediate flags'}

    FORMAT YOUR RESPONSE AS:
    **SUSPICIOUS ACTIVITY:** [One sentence - what happened]

    **KEY CONCERNS:**
    • [Primary red flag]
    • [Secondary red flag]  
    • [Additional concern if applicable]

    **RECOMMENDED ACTION:** [File SAR/Monitor/Escalate - one sentence]

    STRICT LIMITS: 
    - Maximum 100 words
    - No legal disclaimers
    - No background explanations
    - Bullet points only for concerns
    """
    
    def generate_enhanced_sar(self, case: Dict) -> Dict[str, Any]:
        """Generate SAR with comprehensive analysis"""
        
        # Ensure required fields with defaults
        case.setdefault('case_id', f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        case.setdefault('transaction_time', datetime.now().isoformat())
        case.setdefault('currency', 'USD')
        
        # Run comprehensive analysis
        analysis_results = self.analyze_transaction(case)
        
        # Generate enhanced prompt
        prompt = self.format_concise_prompt(case, analysis_results)
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert compliance analyst specializing in financial crime detection and SAR preparation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent compliance language
                max_tokens=800
            )
            
            sar_narrative = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating SAR: {e}")
            sar_narrative = f"Error generating SAR narrative: {str(e)}"
        
        return {
            'sar_narrative': sar_narrative,
            'analysis_results': analysis_results,
            'case_data': case,
            'generation_timestamp': datetime.now().isoformat()
        }
    
    def save_comprehensive_output(self, sar_output: Dict, output_dir: str = "demo"):
        """Save all outputs in organized format"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        case_id = sar_output['case_data']['case_id']
        
        # Save SAR narrative
        with open(f"{output_dir}/sar_narrative_{case_id}.txt", "w") as f:
            f.write(sar_output['sar_narrative'])
        
        # Save detailed analysis
        analysis_summary = {
            'case_id': case_id,
            'risk_assessment': {
                'risk_level': sar_output['analysis_results']['risk'].risk_level,
                'risk_score': sar_output['analysis_results']['risk'].risk_score,
                'risk_factors': sar_output['analysis_results']['risk'].risk_factors,
                'regulatory_flags': sar_output['analysis_results']['risk'].regulatory_flags
            },
            'anomaly_detection': {
                'anomaly_flag': sar_output['analysis_results']['anomaly'].anomaly_flag,
                'anomaly_score': sar_output['analysis_results']['anomaly'].anomaly_score,
                'detected_patterns': sar_output['analysis_results']['anomaly'].detected_patterns
            },
            'model_explanation': {
                'top_features': sar_output['analysis_results']['explanation'].top_features,
                'confidence_score': sar_output['analysis_results']['explanation'].confidence_score,
                'explanation_text': sar_output['analysis_results']['explanation'].explanation_text
            }
        }
        
        with open(f"{output_dir}/analysis_summary_{case_id}.json", "w") as f:
            json.dump(analysis_summary, f, indent=2, default=str)
        
        # Save complete output
        with open(f"{output_dir}/complete_output_{case_id}.json", "w") as f:
            json.dump(sar_output, f, indent=2, default=str)
        
        logger.info(f"All outputs saved to {output_dir}/ with case ID {case_id}")

def main():
    """Example usage of the integrated SAR generation system"""
    
    # Initialize the integrated system
    sar_generator = IntegratedSARGenerator()
    
    # Sample transaction data
    sample_case = {
        "case_id": "CASE_2024_001",
        "from_account": "123456789",
        "from_bank": "First National Bank",
        "to_account": "987654321",
        "to_bank": "International Bank Ltd",
        "amount": 75000,
        "currency": "USD",
        "transaction_time": "2024-01-15T14:30:00Z",
        "recipient_country": "offshore",
        "transaction_frequency": 15,
        "kyc_completeness": 0.6,
        "account_age": 30,
        "beneficiary_type": "corporate",
        "pattern_summary": "Multiple large transactions to offshore accounts",
        "kyc_summary": "KYC documentation incomplete - missing beneficial ownership",
        "customer_notes": "Customer recently increased transaction volume significantly",
        "regulatory_reference": "BSA Section 1020.320 - Suspicious Activity Reporting"
    }
    
    # Generate comprehensive SAR
    logger.info("Starting comprehensive SAR generation...")
    sar_output = sar_generator.generate_enhanced_sar(sample_case)
    
    # Save outputs
    sar_generator.save_comprehensive_output(sar_output)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE SAR GENERATION COMPLETED")
    print("="*60)
    print(f"Case ID: {sar_output['case_data']['case_id']}")
    print(f"Risk Level: {sar_output['analysis_results']['risk'].risk_level}")
    print(f"Risk Score: {sar_output['analysis_results']['risk'].risk_score:.3f}")
    print(f"Anomaly Detected: {sar_output['analysis_results']['anomaly'].anomaly_flag}")
    print(f"Model Confidence: {sar_output['analysis_results']['explanation'].confidence_score:.3f}")
    print("\nFiles generated:")
    print("- sar_narrative_CASE_2024_001.txt")
    print("- analysis_summary_CASE_2024_001.json") 
    print("- complete_output_CASE_2024_001.json")
    print("\nSAR Narrative Preview:")
    print("-" * 40)
    print(sar_output['sar_narrative'][:300] + "...")

if __name__ == "__main__":
    main()