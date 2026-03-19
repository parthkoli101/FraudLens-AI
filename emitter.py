import pandas as pd
import time
import requests
import sys
import os
import random
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TRANSACTIONS_CSV = "upi_transactions.csv"
CUSTOMERS_CSV    = "upi_customers.csv"
MERCHANTS_CSV    = "upi_merchants.csv"
NODE_API_URL     = "http://localhost:5000/api/transactions"
INTERVAL_SECONDS = 5

def run_emitter():
    print("=" * 60)
    print("  FraudLens AI - High-Fidelity Transaction Emitter")
    print("=" * 60)
    print(f"  Streaming data every {INTERVAL_SECONDS} seconds...\n")

    try:
        txns_df  = pd.read_csv(TRANSACTIONS_CSV)
        custs_df = pd.read_csv(CUSTOMERS_CSV)
        merchs_df = pd.read_csv(MERCHANTS_CSV)
        print(f"  Loaded {len(txns_df)} transactions, {len(custs_df)} customers, {len(merchs_df)} merchants.\n")
    except FileNotFoundError as e:
        print(f"ERROR: CSV file not found: {e}")
        sys.exit(1)

    # Merge all three datasets for a fully-enriched view
    df = txns_df.merge(custs_df, on='customer_uuid', how='left')
    df = df.merge(merchs_df, on='merchant_uuid', how='left')
    df = df.sample(frac=1).reset_index(drop=True)
    idx = 0

    while True:
        row = df.iloc[idx % len(df)].to_dict()
        row['transaction_timestamp'] = datetime.now().isoformat()

        # ── Derive real fraud signals ─────────────────────────────────────────
        last_amt = float(row.get('last_transaction_amount', 1000) or 1000)
        amount   = float(row.get('transaction_amount', 0) or 0)
        amt_ratio = round(amount / (last_amt + 0.001), 2)

        new_device = str(row.get('customer_device_id', '')) != str(row.get('registered_device_id', ''))
        new_ip     = str(row.get('customer_ip_address', '')) != str(row.get('registered_ip_address', ''))
        diff_city  = str(row.get('transaction_location', '')) != str(row.get('home_branch', ''))

        # ── Fraud scenario ────────────────────────────────────────────────────
        has_anomaly = random.random() < 0.20
        fraud_prob  = random.random() * 15
        flags       = []

        if has_anomaly:
            scenario = random.randint(0, 3)
            if scenario == 0:
                fraud_prob = 82 + random.random() * 17
                flags = ["Critical Amount Spike", "Unregistered Device"]
            elif scenario == 1:
                fraud_prob = 62 + random.random() * 18
                flags = ["Cross-Border Location Jump", "Anonymous VPN"]
            elif scenario == 2:
                fraud_prob = 42 + random.random() * 18
                flags = ["Inconsistent Purchase Velocity"]
            else:
                fraud_prob = 22 + random.random() * 18
                flags = ["Unusual Merchant Category"]

        # Append real signal flags
        if new_device and "Unregistered Device" not in flags:
            flags.append("Unregistered Device")
        if new_ip and "Unknown IP" not in flags:
            flags.append("Unknown IP")
        if diff_city and "Diff City" not in flags:
            flags.append("Diff City")
        if amt_ratio > 3 and "High Amount Spike" not in flags:
            flags.append("High Amount Spike")

        risk = "LOW"
        if fraud_prob >= 80:   risk = "CRITICAL"
        elif fraud_prob >= 60: risk = "HIGH"
        elif fraud_prob >= 40: risk = "MEDIUM"

        # ── Build payload with ALL customer+merchant fields ───────────────────
        payload = {
            # Transaction
            "transaction_uuid":      row['transaction_uuid'],
            "transaction_amount":    amount,
            "transaction_timestamp": row['transaction_timestamp'],
            "transaction_location":  row.get('transaction_location', ''),
            "customer_device_id":    row.get('customer_device_id', ''),
            "customer_ip_address":   row.get('customer_ip_address', ''),

            # Customer
            "customer_uuid":     row['customer_uuid'],
            "customer_name":     row.get('full_name', 'Unknown'),
            "customer_bank":     row.get('bank_name', 'Unknown Bank'),
            "customer_age":      int(row.get('age', 0) or 0),
            "customer_balance":  float(row.get('account_balance', 0) or 0),
            "customer_last_amt": last_amt,
            "amt_ratio":         amt_ratio,
            "customer_reg_device": row.get('registered_device_id', ''),
            "customer_reg_ip":     row.get('registered_ip_address', ''),
            "customer_txn_count":  int(row.get('total_transactions_count', 0) or 0),
            "customer_upi":      row.get('upi_id', ''),
            "customer_branch":   row.get('home_branch', ''),

            # Merchant (now using real CSV data)
            "merchant_uuid":   row['merchant_uuid'],
            "merchant_name":   row.get('merchant_name', 'Unknown Merchant'),
            "merchant_upi":    row.get('merchant_upi_id', ''),
            "merchant_bank":   row.get('merchant_bank_name', ''),
            "merchant_branch": row.get('merchant_bank_branch', ''),
            "merchant_since":  str(row.get('merchant_account_open_date', ''))[:10],

            # ML / Fraud
            "fraud_probability": round(fraud_prob, 2),
            "prediction":        1 if fraud_prob >= 50 else 0,
            "risk_level":        risk,
            "flags":             flags,
            "new_device":        new_device,
            "new_ip":            new_ip,
            "diff_city":         diff_city,
        }

        try:
            res = requests.post(NODE_API_URL, json=payload, timeout=5)
            if res.status_code in [200, 201]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ PUSHED: {payload['transaction_uuid']} "
                      f"({risk}) - {payload['customer_name']} → {payload['merchant_name']}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ NODE ERROR: {res.status_code}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  API Down: {e}")

        idx += 1
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run_emitter()
