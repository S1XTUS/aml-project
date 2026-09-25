import json
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.risk_classifier import (get_classifier, classify_transaction, transaction_from_case,
                                        normalize_bank, normalize_currency, normalize_payment_format)
from src.models.anomaly_detector import AnomalyDetector as TrainedAnomalyDetector
from src.models.explainer import TransactionExplainer
from src.llm.llm_client import get_llm_client
from src.utils.config_loader import load_config, config_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HIGH_RISK_JURISDICTIONS = {'offshore', 'high-risk', 'high-risk country', 'sanctions'}

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
    time_series_anomalies: List[Dict] = field(default_factory=list)

@dataclass
class RiskClassificationResult:
    """Output from risk_classifier.py"""
    risk_level: str  # LOW, MEDIUM, HIGH
    risk_score: float
    risk_factors: List[str]
    regulatory_flags: List[str]
    compliance_notes: str
    recommendation: str
    is_suspicious: bool = False
    confidence: float = 0.0
    scoring_method: str = "model"


def _rule_based_patterns(case: Dict) -> List[str]:
    """Human-readable red flags from case fields the models do not see."""
    patterns = []
    amount = float(case.get('amount', 0) or 0)
    if amount > 50000:
        patterns.append("Unusually high transaction amount")

    try:
        hour = pd.to_datetime(case.get('transaction_time')).hour
        if hour < 6 or hour > 22:
            patterns.append("Off-hours transaction timing")
    except (ValueError, TypeError):
        pass

    if _is_high_risk_jurisdiction(case):
        patterns.append("High-risk jurisdiction recipient")

    if float(case.get('transaction_frequency', 0) or 0) > 10:
        patterns.append("High frequency transaction pattern")
    return patterns


def _is_high_risk_jurisdiction(case: Dict) -> bool:
    return any(str(case.get(key, '')).lower() in HIGH_RISK_JURISDICTIONS
               for key in ('recipient_country', 'jurisdiction'))


class ModelAnomalyDetector:
    """Runs the trained Isolation Forest from anomaly_detector.py on a single case"""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or os.path.join(config_path("model_dir"), "isolation_forest")
        self._detector = None
        self._load_failed = False

    def _get_detector(self) -> Optional[TrainedAnomalyDetector]:
        if self._detector is None and not self._load_failed:
            try:
                detector = TrainedAnomalyDetector()
                detector.load_model(self.model_dir)
                self._detector = detector
            except Exception as e:
                logger.warning(f"Anomaly model unavailable ({e}); reporting rule-based patterns only")
                self._load_failed = True
        return self._detector

    @staticmethod
    def _case_frame(case: Dict) -> pd.DataFrame:
        """One-row frame in the processed-data schema the anomaly model was trained on."""
        amount = float(case.get('amount', 0) or 0)
        currency = normalize_currency(case.get('currency', 'USD'))
        return pd.DataFrame([{
            'Timestamp': pd.to_datetime(case.get('transaction_time'), errors='coerce'),
            'From_Bank': normalize_bank(case.get('from_bank')),
            'From_Account': str(case.get('from_account', '')),
            'To_Bank': normalize_bank(case.get('to_bank')),
            'To_Account': str(case.get('to_account', '')),
            'Amount_Received': amount,
            'Receiving_Currency': currency,
            'Amount_Paid': amount,
            'Payment_Currency': currency,
            'Payment_Format': normalize_payment_format(case.get('transaction_type', 'Unknown')),
        }])

    def detect_anomalies(self, case: Dict) -> AnomalyResult:
        patterns = _rule_based_patterns(case)
        detector = self._get_detector()

        if detector is None:
            return AnomalyResult(
                anomaly_flag=False,
                anomaly_score=0.0,
                anomaly_type="UNAVAILABLE",
                detected_patterns=patterns,
                baseline_deviation=0.0
            )

        frame = self._case_frame(case)
        result = detector.predict_anomalies(frame)
        anomaly_flag = bool(result['anomaly_flag'].iloc[0])
        # Stored score is "higher = more normal"; flip so higher = more anomalous
        anomaly_score = float(-result['anomaly_score'].iloc[0])

        # Amount deviation from the training population, in standard deviations
        baseline_deviation = 0.0
        if 'Amount_Paid' in detector.feature_names:
            i = detector.feature_names.index('Amount_Paid')
            baseline_deviation = float((frame['Amount_Paid'].iloc[0] - detector.scaler.mean_[i]) / detector.scaler.scale_[i])

        if anomaly_flag:
            patterns.insert(0, "Statistical outlier versus historical transaction population")

        return AnomalyResult(
            anomaly_flag=anomaly_flag,
            anomaly_score=anomaly_score,
            anomaly_type="ANOMALOUS" if anomaly_flag else "NORMAL",
            detected_patterns=patterns,
            baseline_deviation=baseline_deviation
        )


class ModelRiskClassifier:
    """Scores the case with the trained XGBoost risk classifier from risk_classifier.py"""

    RECOMMENDATIONS = {
        'LOW': "Monitor transaction, no immediate action required",
        'MEDIUM': "Enhanced monitoring recommended, review customer profile",
        'HIGH': "Immediate review required, consider filing SAR"
    }

    def classify_risk(self, case: Dict, anomaly_result: AnomalyResult) -> RiskClassificationResult:
        result = classify_transaction(transaction_from_case(case))

        risk_factors = []
        regulatory_flags = []

        if result['is_suspicious']:
            risk_factors.append(f"Risk model flags transaction as suspicious "
                                f"(risk {result['risk_score'] * 100:.1f}% >= threshold {result['threshold_used'] * 100:.1f}%)")

        amount = float(case.get('amount', 0) or 0)
        if amount > 100000:
            risk_factors.append("Large transaction amount exceeds reporting threshold")
            regulatory_flags.append("CTR_REQUIRED")
        elif amount > 10000:
            risk_factors.append("Significant transaction amount")

        if _is_high_risk_jurisdiction(case):
            risk_factors.append("High-risk jurisdiction involvement")
            regulatory_flags.append("SANCTIONS_CHECK")

        if float(case.get('kyc_completeness', 1.0)) < 0.7:
            risk_factors.append("Incomplete KYC documentation")
            regulatory_flags.append("KYC_REVIEW")

        if anomaly_result.anomaly_flag:
            risk_factors.extend(p for p in anomaly_result.detected_patterns if p not in risk_factors)

        method_note = ("XGBoost risk model" if result['scoring_method'] == 'model'
                       else "rule-based fallback (trained model unavailable)")
        compliance_notes = (f"Risk score from {method_note}; "
                            f"{len(risk_factors)} supporting factors identified")

        return RiskClassificationResult(
            risk_level=result['risk_level'],
            risk_score=result['risk_score'],
            risk_factors=risk_factors,
            regulatory_flags=regulatory_flags,
            compliance_notes=compliance_notes,
            recommendation=self.RECOMMENDATIONS[result['risk_level']],
            is_suspicious=result['is_suspicious'],
            confidence=result['confidence'],
            scoring_method=result['scoring_method']
        )


class ModelExplainer:
    """SHAP explanation of the risk classifier's score (explainer.py)"""

    def __init__(self):
        self._explainer = None

    def _get_explainer(self) -> Optional[TransactionExplainer]:
        if self._explainer is None:
            classifier = get_classifier()
            if classifier.model is None and not classifier.load_model():
                return None
            self._explainer = TransactionExplainer(
                classifier.model_path, classifier.scaler_path, feature_names=classifier.feature_columns)
        return self._explainer

    def explain_prediction(self, case: Dict, risk_result: RiskClassificationResult) -> ExplanationResult:
        explainer = self._get_explainer() if risk_result.scoring_method == 'model' else None
        if explainer is None:
            return ExplanationResult(
                top_features=[],
                feature_importance_scores={},
                explanation_text="Model explanation unavailable (rule-based score)",
                confidence_score=risk_result.confidence
            )

        features = get_classifier().build_feature_frame(transaction_from_case(case))
        contributions = explainer.explain_with_shap(features, top_k=len(features.columns))
        top_features = list(contributions.items())[:5]

        drivers = [f"{name} ({'raises' if value > 0 else 'lowers'} risk, {value:+.2f})"
                   for name, value in top_features[:3]]
        return ExplanationResult(
            top_features=top_features,
            feature_importance_scores=contributions,
            explanation_text=f"Primary model drivers (SHAP, log-odds): {', '.join(drivers)}",
            confidence_score=risk_result.confidence
        )


class IntegratedSARGenerator:
    """Main class that combines all three components for SAR generation"""

    def __init__(self):
        self.anomaly_detector = ModelAnomalyDetector()
        self.risk_classifier = ModelRiskClassifier()
        self.explainer = ModelExplainer()

    def analyze_transaction(self, transaction_data: Dict) -> Dict[str, Any]:
        """Run all three components and combine results"""

        logger.info(f"Analyzing transaction {transaction_data.get('case_id', 'Unknown')}")

        anomaly_result = self.anomaly_detector.detect_anomalies(transaction_data)
        logger.info("Anomaly detection completed")

        risk_result = self.risk_classifier.classify_risk(transaction_data, anomaly_result)
        logger.info("Risk classification completed")

        explanation_result = self.explainer.explain_prediction(transaction_data, risk_result)
        logger.info("Explainer analysis completed")

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

    RISK ASSESSMENT: {risk.risk_level} RISK ({risk.risk_score * 100:.1f}% laundering probability)

    RED FLAGS DETECTED:
    {chr(10).join([f'• {flag}' for flag in risk.risk_factors[:4]]) or '• None identified'}

    ANOMALIES: {' | '.join(anomaly.detected_patterns[:2]) if anomaly.detected_patterns else 'Standard pattern'}

    MODEL DRIVERS: {explanation.explanation_text}

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
            llm_config = load_config()["llm"]
            response = get_llm_client().chat.completions.create(
                model=llm_config["model"],
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
        with open(f"{output_dir}/sar_narrative_{case_id}.txt", "w", encoding="utf-8") as f:
            f.write(sar_output['sar_narrative'])

        # Save detailed analysis
        analysis_summary = {
            'case_id': case_id,
            'risk_assessment': {
                'risk_level': sar_output['analysis_results']['risk'].risk_level,
                'risk_score': sar_output['analysis_results']['risk'].risk_score,
                'scoring_method': sar_output['analysis_results']['risk'].scoring_method,
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

        with open(f"{output_dir}/analysis_summary_{case_id}.json", "w", encoding="utf-8") as f:
            json.dump(analysis_summary, f, indent=2, default=str)

        # Save complete output
        with open(f"{output_dir}/complete_output_{case_id}.json", "w", encoding="utf-8") as f:
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
        "transaction_type": "wire_transfer",
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
    print(f"Risk Score: {sar_output['analysis_results']['risk'].risk_score:.3f} "
          f"({sar_output['analysis_results']['risk'].scoring_method})")
    print(f"Anomaly Detected: {sar_output['analysis_results']['anomaly'].anomaly_flag}")
    print(f"Model Confidence: {sar_output['analysis_results']['explanation'].confidence_score:.3f}")
    print(f"Explanation: {sar_output['analysis_results']['explanation'].explanation_text}")
    print("\nFiles generated:")
    print("- sar_narrative_CASE_2024_001.txt")
    print("- analysis_summary_CASE_2024_001.json")
    print("- complete_output_CASE_2024_001.json")
    print("\nSAR Narrative Preview:")
    print("-" * 40)
    print(sar_output['sar_narrative'][:300] + "...")

if __name__ == "__main__":
    main()
