import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import ai_report
import ml_model
import pandas as pd

load_dotenv()

app = FastAPI()

# Load models and data once
model, encoders = ml_model.load_model()
customers_df = pd.read_csv("upi_customers.csv")
merchants_df = pd.read_csv("upi_merchants.csv")

class PredictionRequest(BaseModel):
    transaction: dict

class ReportRequest(BaseModel):
    transaction: dict
    customer: dict
    merchant: dict
    ml_result: dict
    provider: str = "Hugging Face"

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # We need a small mock of the transactions_df or just pass required context
        # In a real scenario, we'd fetch historical context from DB
        # For now, we use a simplified version of ml_model logic
        prob, meta = ml_model.predict_transaction(
            request.transaction, 
            model, 
            encoders, 
            customers_df, 
            merchants_df,
            pd.DataFrame() # No historical context for now in the wrapper
        )
        return {"fraud_probability": prob, **meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_report")
async def generate_report(request: ReportRequest):
    try:
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API Key not found in .env")
        
        report = ai_report.generate_investigation_report(
            request.transaction,
            request.customer,
            request.merchant,
            request.ml_result,
            api_key,
            request.provider
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
