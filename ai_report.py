"""
FraudLens AI - GenAI Report Generation Module
Uses LLM API to generate structured investigation reports.
Falls back to a rich rule-based report when no API key is configured.
"""

import json
import re
import requests
from datetime import datetime


def generate_investigation_report(
    transaction: dict,
    customer: dict,
    merchant: dict,
    ml_result: dict,
    api_key: str,
    provider: str = "Anthropic"
) -> dict:
    """
    Generate a structured investigation report using an LLM API.
    Falls back to _fallback_report() when the API is unavailable/not configured.
    """

    fraud_prob = ml_result.get('fraud_probability', 0)
    flags      = ml_result.get('flags', [])
    prediction = ml_result.get('prediction', 0)

    new_device = ml_result.get('new_device', 0)
    new_ip     = ml_result.get('new_ip', 0)
    diff_city  = ml_result.get('diff_city', 0)
    amt_ratio  = ml_result.get('amt_ratio', ml_result.get('txn_amount_ratio', 1))

    cust_name    = customer.get('full_name', 'N/A')
    cust_bank    = customer.get('bank_name', 'N/A')
    cust_branch  = customer.get('home_branch', 'N/A')
    cust_bal     = float(customer.get('account_balance', 0) or 0)
    cust_age     = customer.get('age', 'N/A')
    cust_txns    = customer.get('total_transactions_count', 'N/A')
    cust_last    = float(customer.get('last_transaction_amount', 0) or 0)
    reg_device   = customer.get('registered_device_id', 'N/A')
    reg_ip       = customer.get('registered_ip_address', 'N/A')
    cust_upi     = customer.get('upi_id', 'N/A')

    merch_name   = merchant.get('merchant_name', 'N/A')
    merch_bank   = merchant.get('merchant_bank_name', 'N/A')
    merch_branch = merchant.get('merchant_bank_branch', 'N/A')
    merch_upi    = merchant.get('merchant_upi_id', 'N/A')
    merch_since  = merchant.get('merchant_account_open_date', 'N/A')

    txn_city  = transaction.get('transaction_location', 'N/A')
    txn_dev   = transaction.get('customer_device_id', 'N/A')
    txn_ip    = transaction.get('customer_ip_address', 'N/A')
    txn_amt   = float(transaction.get('transaction_amount', 0) or 0)
    txn_ts    = str(transaction.get('transaction_timestamp', 'N/A'))[:19]

    amount_spike_str = f"YES — {amt_ratio:.1f}x above customer average" if amt_ratio > 1.5 else "NO — amount within normal range"

    prompt = f"""You are a senior bank fraud investigation officer at an Indian commercial bank. Analyze the following UPI transaction and produce a HIGH-FIDELITY investigation report.

CRITICAL: Provide a deep, specific explanation of WHY this transaction is fraudulent or legitimate. Reference the ACTUAL data values below. Do NOT give generic responses.

## Customer Profile
- Name: {cust_name} | Age: {cust_age} | Bank: {cust_bank}
- Home Branch: {cust_branch} | UPI ID: {cust_upi}
- Account Balance: Rs {cust_bal:,.2f} | Last Txn Amount: Rs {cust_last:,.2f}
- Total Historical Transactions: {cust_txns}
- Registered Device: {reg_device}
- Registered IP: {reg_ip}

## Transaction Data
- ID: {transaction.get('transaction_uuid')}
- Amount: Rs {txn_amt:,.2f} ({amt_ratio:.2f}x customer average)
- Timestamp: {txn_ts}
- Location: {txn_city}
- Device Used: {txn_dev}
- IP Address Used: {txn_ip}
- Anomaly Flags: {', '.join(flags) if flags else 'None'}

## Merchant Profile
- Name: {merch_name} | Bank: {merch_bank} | Branch: {merch_branch}
- Merchant UPI: {merch_upi} | Account Since: {merch_since}

## ML Model Risk Assessment
- Fraud Probability: {fraud_prob:.1f}%
- Model Verdict: {'FRAUDULENT' if prediction == 1 else 'LEGITIMATE'}
- Device Anomaly: {'YES — used device does not match registered device' if new_device else 'NO — device matches registered device'}
- IP Anomaly: {'YES — IP does not match registered IP' if new_ip else 'NO — IP matches registered IP'}
- City Anomaly: {'YES — transaction city differs from home branch' if diff_city else 'NO — city matches home branch'}
- Amount Spike: {amount_spike_str}

Generate a JSON report matching this structure exactly:
{{
  "investigation_outcome": "FRAUD_CONFIRMED" | "SUSPICIOUS" | "CLEARED",
  "risk_level": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
  "confidence_score": <number 0-100>,
  "executive_summary": "<2-3 sentences summarising the specific risk signals and initial verdict for THIS EXACT transaction. Reference specific values.>",
  "data_analyzed": {{
    "customer_risk_factors": ["<specific customer-related observations referencing real values>"],
    "transaction_anomalies": ["<specific transaction anomalies referencing real values>"],
    "merchant_risk_factors": ["<specific merchant-related observations>"]
  }},
  "detected_inconsistencies": ["<list of specific, data-grounded inconsistencies found>"],
  "reasoning": "<Deep 4-paragraph analysis. P1: what triggered the alert. P2: device/IP/location signals with exact values. P3: amount vs. customer history and balance context. P4: final verdict — WHY fraud/suspicious/cleared with specific evidence.>",
  "recommended_action": "<Specific next step mentioning customer name and transaction ID>",
  "supporting_evidence": ["<Technical proofs grounded in actual data values>"],
  "mitigating_factors": ["<Factors reducing suspicion, or empty array if fraud confirmed>"]
}}
Return ONLY valid JSON. No markdown fences, no text outside the JSON."""

    fallback_args = (transaction, customer, merchant, ml_result, flags, fraud_prob, prediction,
                     new_device, new_ip, diff_city, amt_ratio)
    if not api_key:
        return _fallback_report(*fallback_args)

    try:
        content = None

        if provider == "Anthropic":
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 2500,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            if response.status_code == 200:
                content = response.json()['content'][0]['text']

        elif provider == "OpenAI":
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4-turbo-preview",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                },
                timeout=30
            )
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']

        elif provider == "Hugging Face":
            hf_in = "[INST] " + prompt + " [/INST]"
            r = requests.post(
                "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
                headers={"Authorization": "Bearer " + api_key},
                json={"inputs": hf_in, "parameters": {"max_new_tokens": 1800, "return_full_text": False}},
                timeout=60
            )
            if r.status_code == 200:
                content = r.json()[0]["generated_text"]
            else:
                print("HF ERR:", r.status_code, r.text[:200])

        if content is None:
            return _fallback_report(*fallback_args)

        content = content.strip()
        m = re.search(r'\{[\s\S]*\}', content)
        if m:
            content = m.group(0)
        elif content.startswith("```"):
            parts = content.split("```")
            content = parts[1].lstrip("json").strip() if len(parts) > 1 else content

        report = json.loads(content)
        report["generated_at"]   = datetime.now().isoformat()
        report["transaction_id"] = transaction.get("transaction_uuid")
        return report

    except Exception as e:
        print("AI REPORT ERROR:", type(e).__name__, e)
        return _fallback_report(*fallback_args)


def _fallback_report(transaction, customer, merchant, ml_result, flags, fraud_prob, prediction,
                     new_device, new_ip, diff_city, amt_ratio):
    """Rich rule-based report using real customer and merchant data."""
    outcome = "FRAUD_CONFIRMED" if fraud_prob > 70 else ("SUSPICIOUS" if fraud_prob > 40 else "CLEARED")
    risk    = "CRITICAL" if fraud_prob > 80 else ("HIGH" if fraud_prob > 60 else ("MEDIUM" if fraud_prob > 40 else "LOW"))

    cust_name   = customer.get("full_name", "the customer")
    cust_bank   = customer.get("bank_name", "their bank")
    cust_branch = customer.get("home_branch", "home branch")
    cust_bal    = float(customer.get("account_balance", 0) or 0)
    cust_txns   = int(customer.get("total_transactions_count", 0) or 0)
    cust_last   = float(customer.get("last_transaction_amount", 0) or 0)
    reg_device  = customer.get("registered_device_id", "N/A")
    reg_ip      = customer.get("registered_ip_address", "N/A")

    merch_name  = merchant.get("merchant_name", "the merchant")
    merch_bank  = merchant.get("merchant_bank_name", "")

    txn_id   = transaction.get("transaction_uuid", "UNKNOWN")
    txn_city = transaction.get("transaction_location", "N/A")
    txn_dev  = transaction.get("customer_device_id", "N/A")
    txn_ip   = transaction.get("customer_ip_address", "N/A")
    txn_amt  = float(transaction.get("transaction_amount", 0) or 0)
    amt_ratio = float(amt_ratio or 1)

    # Build specific, data-grounded fraud reasons
    fraud_reasons = []
    if new_device:
        fraud_reasons.append('Device mismatch: used "{}" — registered device is "{}"'.format(txn_dev, reg_device))
    if new_ip:
        fraud_reasons.append('IP mismatch: used "{}" — registered IP is "{}"'.format(txn_ip, reg_ip))
    if diff_city:
        fraud_reasons.append('Location anomaly: transaction in "{}" but customer home branch is "{}"'.format(txn_city, cust_branch))
    if amt_ratio > 1.5:
        fraud_reasons.append('Amount spike: Rs {:.2f} is {:.1f}x above customer avg of Rs {:.2f}'.format(txn_amt, amt_ratio, cust_last))
    for f in flags:
        if f not in ["Unregistered Device", "Unknown IP", "Diff City", "High Amount Spike"]:
            fraud_reasons.append(f)

    if outcome == "FRAUD_CONFIRMED":
        exec_summary = (
            '{} transaction of Rs {:,.2f} to {} is CONFIRMED FRAUDULENT. '
            '{} critical signal(s) detected: {}.'.format(
                cust_name, txn_amt, merch_name, len(fraud_reasons),
                fraud_reasons[0] if fraud_reasons else (flags[0] if flags else "multiple anomalies")))
        reasoning = (
            'Transaction {} was flagged at {:.1f}% fraud probability with {} anomaly signals detected.\n\n'.format(txn_id, fraud_prob, len(flags))
            + ('Critical signals: {}. '.format('; '.join(fraud_reasons)) if fraud_reasons else '')
            + 'The convergence of multiple high-severity signals strongly indicates account compromise or unauthorized access.\n\n'
            + '{} ({}, home branch: {}) holds a balance of Rs {:,.2f}. '.format(cust_name, cust_bank, cust_branch, cust_bal)
            + 'The attempted transfer of Rs {:,.2f} to {}{}  is {:.1f}x the customer historical average of Rs {:,.2f}.\n\n'.format(
                txn_amt, merch_name, ' ({})'.format(merch_bank) if merch_bank else '', amt_ratio, cust_last)
            + 'Immediate intervention is required. The pattern is consistent with an account takeover attack.')
        rec = 'Block transaction {} immediately. Freeze account and contact {} on their registered mobile. Escalate if unresponsive within 2 hours.'.format(txn_id, cust_name)
        mit = []

    elif outcome == "SUSPICIOUS":
        exec_summary = (
            '{} Rs {:,.2f} transaction to {} is SUSPICIOUS ({:.1f}% risk). '
            'Manual verification required.'.format(cust_name, txn_amt, merch_name, fraud_prob))
        reasoning = (
            'Transaction {} raised {} moderate anomaly signals at {:.1f}% fraud probability.\n\n'.format(txn_id, len(flags), fraud_prob)
            + ('Notable: {}. '.format('; '.join(fraud_reasons)) if fraud_reasons else '')
            + 'While not conclusive alone, the combination warrants human verification before processing.\n\n'
            + '{} ({}) has {} historical transactions — indicating an established account. '.format(cust_name, cust_bank, cust_txns)
            + 'Amount Rs {:,.2f} is {:.1f}x above the customer average of Rs {:,.2f}.\n\n'.format(txn_amt, amt_ratio, cust_last)
            + 'A quick verification call to the registered contact should resolve this case.')
        rec = 'Place temporary hold on transaction {}. Contact {} via registered contact to verify intent. Clear if authenticated within SLA.'.format(txn_id, cust_name)
        mit = ['Established account with {} prior transactions'.format(cust_txns)] if cust_txns > 50 else []

    else:
        exec_summary = (
            '{} Rs {:,.2f} transaction to {} is CLEARED. '
            'All telemetry (device, IP, location, amount) aligns with the customer established profile.'.format(cust_name, txn_amt, merch_name))
        reasoning = (
            'Transaction {} scored only {:.1f}% fraud probability — well below the investigation threshold.\n\n'.format(txn_id, fraud_prob)
            + 'Device and IP address match {}\'s registered profile. '.format(cust_name)
            + 'The transaction city "{}" is consistent with home branch "{}". No geographic anomaly detected.\n\n'.format(txn_city, cust_branch)
            + 'The amount Rs {:,.2f} represents a {:.2f}x ratio vs. customer average Rs {:,.2f} — '.format(txn_amt, amt_ratio, cust_last)
            + 'within acceptable variance for this account segment.\n\n'
            + 'With {} historical transactions on record for this {} customer, '.format(cust_txns, cust_bank)
            + 'the behavioural pattern is well established and consistent. No action required.')
        rec = 'Approve and clear transaction {} from the investigation queue. No further action needed.'.format(txn_id)
        mit = [
            'Device and IP match registered profile — no compromise detected',
            'Location "{}" is consistent with home branch "{}"'.format(txn_city, cust_branch),
            'Amount ratio {:.2f}x is within acceptable range'.format(amt_ratio),
            'Established account with {} prior transactions'.format(cust_txns),
        ]

    customer_risk = [r for r in fraud_reasons if "Device" in r or "IP" in r] or ["No device/IP anomalies detected"]
    txn_anomalies = [r for r in fraud_reasons if "Amount" in r or "amount" in r.lower()] or ["Transaction volume within normal range"]
    merch_risk    = [f for f in flags if "merchant" in f.lower()] or ["Standard merchant — no elevated risk flags on record"]

    return {
        "investigation_outcome":    outcome,
        "risk_level":               risk,
        "confidence_score":         round(fraud_prob),
        "executive_summary":        exec_summary,
        "data_analyzed": {
            "customer_risk_factors":  customer_risk,
            "transaction_anomalies":  txn_anomalies,
            "merchant_risk_factors":  merch_risk,
        },
        "detected_inconsistencies": fraud_reasons if fraud_reasons else ["None detected — all parameters align with customer profile"],
        "reasoning":                reasoning,
        "recommended_action":       rec,
        "supporting_evidence":      flags if flags else ["All telemetry (device, IP, location, amount) within normal bounds"],
        "mitigating_factors":       mit,
        "generated_at":             datetime.now().isoformat(),
        "transaction_id":           txn_id,
        "_fallback":                True,
    }

