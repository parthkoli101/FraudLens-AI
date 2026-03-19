# 🔍 FraudLens AI
## Financial Intelligence & Investigation AI System

> Intelligent UPI Transaction Investigation powered by XGBoost + Claude AI

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard
```bash
streamlit run app.py
```
Open: **http://localhost:8501**

### 3. (Optional) Start Live Transaction Emitter
In a **separate terminal**:
```bash
python emitter.py
```
This streams one transaction every 5 seconds into the Live Feed tab.

---

## 🔑 API Key Setup
1. Get your Anthropic API key from https://console.anthropic.com
2. Enter it in the **sidebar** of the dashboard under "Anthropic API Key"
3. The key enables AI-powered investigation reports using Claude Sonnet

> Without an API key, the system uses rule-based fallback reports.

---

## 📁 File Structure
```
fraudlens/
├── app.py                  # Main Streamlit dashboard
├── ml_model.py             # XGBoost fraud detection model
├── ai_report.py            # Claude AI report generation
├── datastore.py            # SQLite audit trail & live feed
├── emitter.py              # Real-time transaction simulator
├── requirements.txt        # Python dependencies
├── upi_customers.csv       # Customer dataset
├── upi_merchants.csv       # Merchant dataset
└── upi_transactions.csv    # Transaction dataset
```

---

## 🎯 Features

### Command Center
- Live KPI metrics (total transactions, flagged cases, fraud rate)
- Fraud probability distribution chart
- City-wise flagged transaction heatmap
- Risk level breakdown (CRITICAL / HIGH / MEDIUM / LOW)
- One-click investigation launch from flagged list

### Investigation Panel
- Full transaction, customer, and merchant data display
- XGBoost fraud probability score with gauge visualization
- Anomaly flag highlighting (new device, new IP, location mismatch, amount spike)
- Customer transaction history chart
- **AI-generated investigation report** (Claude AI):
  - Investigation outcome
  - Detected inconsistencies
  - AI reasoning (3-4 paragraph)
  - Recommended action
- Officer decision buttons: CONFIRM FRAUD / CLEAR / ESCALATE
- Auto-saves to audit trail

### Analytics Hub
- Transaction amount by city
- Customer distribution by bank
- Fraud risk by age group
- Top merchants by fraud exposure
- Amount vs. fraud probability scatter

### Audit Trail
- Complete log of all officer decisions
- Compliance-ready record with timestamps
- Decision summary statistics

### Live Feed (Real-Time Simulation)
- Streams transactions from `emitter.py`
- Auto-detects high-risk transactions
- Visual FRAUD ALERT banners
- Auto-refresh toggle (5s interval)

---

## 🧠 ML Model Details

**Algorithm**: XGBoost Classifier  
**Balancing**: SMOTE oversampling  
**Features**: 
- Account balance, transaction amount, amount ratio
- Device/IP mismatch flags
- Location mismatch (home branch vs transaction city)
- Transaction velocity, merchant diversity

**Fraud Label Logic** (rule-based for dataset generation):
- Amount > 1.5x customer's last transaction
- Different city from home branch
- Unregistered device used
- Unknown IP address

---

## 🏆 Hackathon Demo Tips

1. **Start with Command Center** — show the KPI overview
2. **Click "Investigate →"** on a CRITICAL risk transaction
3. **Hit "Generate AI Investigation Report"** — wow the judges
4. **Make a decision** — show the audit trail populating
5. **Open Live Feed** + run `emitter.py` in another terminal — demonstrate real-time event-driven capability
