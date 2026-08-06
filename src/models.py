# src/models.py
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def split_data(df, feature_cols, target_col='fraud_bool'):
    """Performs a stratified split to maintain class distribution."""
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def tune_and_train_logistic_regression(X_train, y_train, preprocessor):
    """
    Builds a pipeline, executes a grid search optimizing for average precision,
    and handles solver convergence constraints via liblinear.
    """
    # 1. Build the full Scikit-learn Pipeline
    log_reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])

    # 2. Define your optimized parameter grid
    param_grid = {
        'model__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
        'model__penalty': ['l2'],
        'model__solver': ['liblinear']
    }

    # 3. Initialize and fit the cross-validated grid search
    cross = GridSearchCV(log_reg_pipeline, param_grid, cv=5, scoring='average_precision', n_jobs=2)
    cross.fit(X_train, y_train)

    # Pack up the training insights into a clean dictionary
    train_insights = {
        'best_params': cross.best_params_,
        'best_cv_score': cross.best_score_
    }

    # Return BOTH the best model and the metrics dictionary
    return cross.best_estimator_, train_insights