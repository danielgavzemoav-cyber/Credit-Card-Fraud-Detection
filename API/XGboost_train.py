import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

print("Loading data...")
df = pd.read_csv('../data/Base.csv')

# Feature engineering
df['balance_to_income_ratio'] = df['intended_balcon_amount'] / df['income']
df['balance_to_income_ratio'] = df['balance_to_income_ratio'].replace([np.inf, -np.inf], np.nan)

# Sentinel values
sentinel_cols = ['current_address_months_count', 'bank_months_count',
                 'session_length_in_minutes', 'device_distinct_emails_8w']
df[sentinel_cols] = df[sentinel_cols].replace(-1, np.nan)

# Column definitions
num_cols = ['income', 'name_email_similarity',
            'current_address_months_count', 'customer_age', 'days_since_request',
            'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h',
            'velocity_4w', 'bank_months_count', 'bank_branch_count_8w',
            'proposed_credit_limit', 'date_of_birth_distinct_emails_4w',
            'session_length_in_minutes', 'balance_to_income_ratio',
            'device_distinct_emails_8w']

cat_cols = ['payment_type', 'source', 'employment_status', 'device_os', 'housing_status']
binary_cols = ['phone_home_valid', 'phone_mobile_valid', 'email_is_free',
               'foreign_request', 'has_other_cards', 'keep_alive_session']

# Preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols),
    ('bin', binary_transformer, binary_cols)
])

# Train/test split
X = df[binary_cols + num_cols + cat_cols]
y = df['fraud_bool']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost pipeline
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.1f}")

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    ))
])

print("Training XGBoost...")
xgb_pipeline.fit(X_train, y_train)

# Evaluate
y_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
ap = average_precision_score(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
print(f"XGBoost — AP: {ap:.4f} | ROC-AUC: {auc:.4f}")

# Save model + column info
joblib.dump({
    'pipeline': xgb_pipeline,
    'binary_cols': binary_cols,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'sentinel_cols': sentinel_cols
}, '../API/fraud_model.joblib')

print("Model saved to fraud_model.joblib")
