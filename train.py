import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

os.makedirs('models', exist_ok=True)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = [
    'status_of_existing_checking_account', 'duration_in_month', 'credit_history', 'purpose',
    'credit_amount', 'savings_account_bonds', 'present_employment_since', 'installment_rate_in_percentage_of_disposable_income',
    'personal_status_and_sex', 'other_debtors_guarantors', 'present_residence_since', 'property',
    'age_in_years', 'other_installment_plans', 'housing', 'number_of_existing_credits_at_this_bank',
    'job', 'number_of_people_being_liable_to_provide_maintenance_for', 'telephone', 'foreign_worker',
    'credit_risk'
]
df = pd.read_csv(url, sep=' ', header=None, names=column_names)

df['credit_risk'] = df['credit_risk'].map({1: 0, 2: 1})

df['credit_amount'] = np.log1p(df['credit_amount'])

df_processed = pd.get_dummies(df, drop_first=True, dtype=int)
X = df_processed.drop('credit_risk', axis=1)
y = df_processed['credit_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numerical_features = ['duration_in_month', 'credit_amount', 'age_in_years', 
                      'present_residence_since', 'number_of_existing_credits_at_this_bank',
                      'number_of_people_being_liable_to_provide_maintenance_for']

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])

model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'models/model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(X_train.columns.tolist(), 'models/columns.joblib')