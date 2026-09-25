import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.risk_classifier import classify_transaction


def transaction_from_row(tx) -> dict:
    """Map a raw IBM AML row ('Amount Paid', 'From Bank', ...) to a scoring transaction."""
    return {
        'timestamp': tx["Timestamp"],
        'from_bank': tx["From Bank"],
        'from_account': tx["Account"],
        'to_bank': tx["To Bank"],
        'to_account': tx["Account.1"],
        'amount_paid': float(tx["Amount Paid"]),
        'amount_received': float(tx["Amount Received"]),
        'receiving_currency': tx["Receiving Currency"],
        'payment_currency': tx["Payment Currency"],
        'payment_format': tx["Payment Format"],
    }


def score_transaction(tx) -> dict:
    """Score a raw transaction row with the trained risk classifier."""
    result = classify_transaction(transaction_from_row(tx))
    return {
        "score": result["risk_score"],
        "is_suspicious": result["is_suspicious"],
        "risk_level": result["risk_level"]
    }


if __name__ == "__main__":
    test_tx = {
        "Timestamp": "2022/09/01 03:15",
        "From Bank": 70,
        "Account": "100428660",
        "To Bank": 1124,
        "Account.1": "800825340",
        "Amount Paid": 80000,
        "Amount Received": 79990,
        "Receiving Currency": "US Dollar",
        "Payment Currency": "Bitcoin",
        "Payment Format": "Bitcoin"
    }

    print(score_transaction(test_tx))
