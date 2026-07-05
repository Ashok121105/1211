import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
from preprocess import load_and_preprocess_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_prob)
        
    print(confusion_matrix(y_test, y_pred))
    return metrics

def train_and_evaluate():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    best_model = None
    best_f1 = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        print(f"Evaluating {name}...")
        metrics = evaluate_model(model, X_test, y_test)
        
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        if metrics['F1-Score'] > best_f1:
            best_f1 = metrics['F1-Score']
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with F1-Score: {best_f1:.4f}")
    
    # Save the best model
    os.makedirs('model', exist_ok=True)
    model_path = 'model/model.pkl'
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
