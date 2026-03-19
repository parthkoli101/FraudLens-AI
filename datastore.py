"""
FraudLens AI - Data Store Module
Manages in-memory data with SQLite persistence for audit trail.
"""

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

DB_PATH = "fraudlens.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL,
        officer_decision TEXT NOT NULL,
        investigation_outcome TEXT,
        risk_level TEXT,
        fraud_probability REAL,
        notes TEXT,
        decided_at TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS live_transactions (
        transaction_uuid TEXT PRIMARY KEY,
        customer_uuid TEXT,
        merchant_uuid TEXT,
        transaction_amount REAL,
        transaction_timestamp TEXT,
        transaction_location TEXT,
        customer_device_id TEXT,
        customer_ip_address TEXT,
        inserted_at TEXT,
        is_processed INTEGER DEFAULT 0
    )''')
    conn.commit()
    conn.close()


def save_audit_decision(transaction_id: str, decision: str, outcome: str, risk: str, prob: float, notes: str = ""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO audit_log 
        (transaction_id, officer_decision, investigation_outcome, risk_level, fraud_probability, notes, decided_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (transaction_id, decision, outcome, risk, prob, notes, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_audit_log() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM audit_log ORDER BY decided_at DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df


def insert_live_transaction(txn: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO live_transactions 
        (transaction_uuid, customer_uuid, merchant_uuid, transaction_amount, 
         transaction_timestamp, transaction_location, customer_device_id, customer_ip_address, inserted_at, is_processed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)''',
        (txn.get('transaction_uuid'), txn.get('customer_uuid'), txn.get('merchant_uuid'),
         txn.get('transaction_amount'), str(txn.get('transaction_timestamp')),
         txn.get('transaction_location'), txn.get('customer_device_id'),
         txn.get('customer_ip_address'), datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_live_transactions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM live_transactions ORDER BY inserted_at DESC LIMIT 50", conn
        )
    except:
        df = pd.DataFrame()
    conn.close()
    return df


def get_live_transaction_count() -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(*) FROM live_transactions")
        count = c.fetchone()[0]
    except:
        count = 0
    conn.close()
    return count
