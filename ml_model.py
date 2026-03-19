"""
FraudLens AI - ML Model Module
Trains and runs XGBoost fraud detection on UPI transaction data.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import os

MODEL_PATH = "fraud_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"

CAT_COLS = ['transaction_location', 'bank_name', 'home_branch',
            'merchant_bank_name', 'merchant_bank_branch']
DEVICE_IP_COLS = ['registered_device_id', 'customer_device_id',
                  'registered_ip_address', 'customer_ip_address']
FEATURES = ['age', 'account_balance', 'last_transaction_amount', 'total_transactions_count',
            'total_transactions_amount', 'transaction_amount', 'txn_amount_ratio',
            'time_since_last_txn', 'num_merchants', 'is_different_city',
            'new_device', 'new_ip'] + CAT_COLS


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
    df['account_open_date'] = pd.to_datetime(df['account_open_date'])
    df['merchant_account_open_date'] = pd.to_datetime(df['merchant_account_open_date'])

    df['avg_txn_amount'] = df.groupby('customer_uuid')['transaction_amount'].transform('mean')
    df['txn_amount_ratio'] = df['transaction_amount'] / (df['avg_txn_amount'] + 1e-5)
    df = df.sort_values(['customer_uuid', 'transaction_timestamp'])
    df['time_since_last_txn'] = df.groupby('customer_uuid')['transaction_timestamp'].diff().dt.total_seconds()
    df['time_since_last_txn'] = df['time_since_last_txn'].fillna(0)
    df['num_merchants'] = df.groupby('customer_uuid')['merchant_uuid'].transform('nunique')
    df['is_different_city'] = (df['transaction_location'] != df['home_branch']).astype(int)
    df['new_device'] = (df['customer_device_id'] != df['registered_device_id']).astype(int)
    df['new_ip'] = (df['customer_ip_address'] != df['registered_ip_address']).astype(int)

    # Aggressive synthetic fraud labeling for training diversity
    df['is_fraud'] = (
        (df['txn_amount_ratio'] > 2.5) |  # Very high amount vs average
        ((df['is_different_city'] == 1) & (df['txn_amount_ratio'] > 1.8)) | # City jump + high amount
        ((df['new_device'] == 1) & (df['new_ip'] == 1)) | # New device AND new IP
        ((df['transaction_amount'] > 50000) & (df['account_balance'] < 100000)) # High value vs balance
    ).astype(int)

    # Ensure we have at least some fraud cases if none were caught
    if df['is_fraud'].sum() < 50: # Aim for ~5% fraud in 1000 items
        fraud_idx = np.random.choice(df.index, size=50, replace=False)
        df.loc[fraud_idx, 'is_fraud'] = 1

    return df


def train_model(customers_df, merchants_df, transactions_df):
    data = transactions_df.merge(customers_df, on='customer_uuid', how='left')
    data = data.merge(merchants_df, on='merchant_uuid', how='left')
    data = build_features(data)

    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le
    for col in DEVICE_IP_COLS:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    X = data[FEATURES]
    y = data['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    except Exception:
        X_train_bal, y_train_bal = X_train, y_train

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric='logloss', random_state=42, use_label_encoder=False
    )
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(encoders, f)

    return model, encoders, acc, data


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders


def predict_transaction(txn_row: dict, model, encoders, customers_df, merchants_df, transactions_df):
    """Given a single transaction dict, compute fraud probability."""
    cust = customers_df[customers_df['customer_uuid'] == txn_row['customer_uuid']]
    merch = merchants_df[merchants_df['merchant_uuid'] == txn_row['merchant_uuid']]

    if cust.empty or merch.empty:
        return None, {}

    row = {**txn_row, **cust.iloc[0].to_dict(), **merch.iloc[0].to_dict()}
    df = pd.DataFrame([row])

    # Derived features using full transaction history for context
    cust_history = transactions_df[transactions_df['customer_uuid'] == txn_row['customer_uuid']]
    avg_amount = cust_history['transaction_amount'].mean() if len(cust_history) > 0 else txn_row['transaction_amount']
    num_merchants = cust_history['merchant_uuid'].nunique() if len(cust_history) > 0 else 1

    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
    df['account_open_date'] = pd.to_datetime(df['account_open_date'])
    df['merchant_account_open_date'] = pd.to_datetime(df['merchant_account_open_date'])
    df['avg_txn_amount'] = avg_amount
    df['txn_amount_ratio'] = df['transaction_amount'] / (avg_amount + 1e-5)
    df['time_since_last_txn'] = 0.0
    df['num_merchants'] = num_merchants
    df['is_different_city'] = int(str(df['transaction_location'].iloc[0]) != str(df['home_branch'].iloc[0]))
    df['new_device'] = int(str(df['customer_device_id'].iloc[0]) != str(df['registered_device_id'].iloc[0]))
    df['new_ip'] = int(str(df['customer_ip_address'].iloc[0]) != str(df['registered_ip_address'].iloc[0]))

    for col in CAT_COLS + DEVICE_IP_COLS:
        le = encoders.get(col)
        if le:
            val = df[col].astype(str).iloc[0]
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = 0
        else:
            df[col] = 0

    X = df[FEATURES]
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    flags = []
    if df['new_device'].iloc[0] == 1:
        flags.append("Unregistered device used")
    if df['new_ip'].iloc[0] == 1:
        flags.append("New/unknown IP address")
    if df['is_different_city'].iloc[0] == 1:
        flags.append("Transaction location differs from home branch")
    if df['txn_amount_ratio'].iloc[0] > 1.5:
        flags.append(f"Amount {df['txn_amount_ratio'].iloc[0]:.1f}x above customer average")

    return prob, {
        'prediction': pred,
        'fraud_probability': round(float(prob) * 100, 2),
        'flags': flags,
        'txn_amount_ratio': round(float(df['txn_amount_ratio'].iloc[0]), 2),
        'is_different_city': int(df['is_different_city'].iloc[0]),
        'new_device': int(df['new_device'].iloc[0]),
        'new_ip': int(df['new_ip'].iloc[0]),
        'avg_amount': round(float(avg_amount), 2),
    }
