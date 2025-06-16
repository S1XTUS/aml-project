import joblib
import numpy as np

# Load the trained model from Phase 2
model = joblib.load("models/risk_classifier_xgb.pkl")

# Optional: define same preprocessing steps used during training
def preprocess(tx: dict) -> np.ndarray:
    # Use only the features used during training
    features = [
        float(tx["Amount Paid"]),
        float(tx["Amount Received"]),
        1 if tx["Receiving Currency"] != tx["Payment Currency"] else 0,
        1 if tx["Payment Format"] in ["Crypto", "Wire"] else 0,
    ]
    return np.array(features).reshape(1, -1)

def score_transaction(tx: dict) -> dict:
    features = preprocess(tx)
    risk_score = model.predict_proba(features)[0][1]  # Probability of 'risky'
    is_suspicious = risk_score > 0.8  # Adjust threshold as needed

    return {
        "score": risk_score,
        "is_suspicious": is_suspicious
    }

if __name__ == "__main__":
    test_tx = {
        "Amount Paid": 80000,
        "Amount Received": 79990,
        "Receiving Currency": "USD",
        "Payment Currency": "BTC",
        "Payment Format": "Crypto"
    }

    print(score_transaction(test_tx))
