import requests
import json

payload = {
    "transaction": {"transaction_uuid": "TXN_TEST123", "transaction_amount": 25000, "transaction_location": "Delhi"},
    "customer": {"customer_uuid": "CUST_TEST456"},
    "merchant": {"merchant_uuid": "MERCH_TEST789"},
    "ml_result": {"fraud_probability": 85.5, "flags": ["Location Jump", "Device Not Recognized"]},
    "provider": "Hugging Face"
}

try:
    print("Sending request to /generate_report...")
    res = requests.post("http://localhost:8000/generate_report", json=payload, timeout=20)
    print("Status:", res.status_code)
    try:
        report = res.json()
        print("\nOutcome:", report.get('investigation_outcome'))
        print("Reasoning:\n", report.get('reasoning'))
        if report.get('_fallback'):
            print("\nWARNING: This is a fallback report.")
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw response:", res.text)
except Exception as e:
    print("Connection error:", e)
