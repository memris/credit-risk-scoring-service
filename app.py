from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Credit scoring API")

try:
    model = joblib.load('models/model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    model_columns = joblib.load('models/columns.joblib')
except Exception as e:
    print(f"error: {e}")

class BorrowerData(BaseModel):
    status_of_existing_checking_account: str
    duration_in_month: int
    credit_history: str
    purpose: str
    credit_amount: float
    savings_account_bonds: str
    present_employment_since: str
    installment_rate_in_percentage_of_disposable_income: int
    personal_status_and_sex: str
    other_debtors_guarantors: str
    present_residence_since: int
    property: str
    age_in_years: int
    other_installment_plans: str
    housing: str
    number_of_existing_credits_at_this_bank: int
    job: str
    number_of_people_being_liable_to_provide_maintenance_for: int
    telephone: str
    foreign_worker: str

@app.post("/predict")
def predict(data: BorrowerData):
    input_df = pd.DataFrame([data.dict()])
    
    input_df['credit_amount'] = np.log1p(input_df['credit_amount'])
    
    input_df_ohe = pd.get_dummies(input_df)

    final_df = pd.DataFrame(columns=model_columns)
    for col in model_columns:
        if col in input_df_ohe.columns:
            final_df[col] = input_df_ohe[col]
        else:
            final_df[col] = 0
    
    num_cols = ['duration_in_month', 'credit_amount', 'age_in_years', 
                'present_residence_since', 'number_of_existing_credits_at_this_bank',
                'number_of_people_being_liable_to_provide_maintenance_for']
    final_df[num_cols] = scaler.transform(final_df[num_cols])
    
    prediction = model.predict(final_df)[0]
    probability = model.predict_proba(final_df)[0][1]
    
    status = "дефолт" if prediction == 1 else "не дефолт"
    
    return {
        "prediction": int(prediction),
        "status": status,
        "probability_of_default": round(float(probability), 4)
    }

@app.get("/")
def read_root():
    return {"message": "Credit scoring API is running"}