import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(filepath='dataset/credit_card_data.csv'):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop('Approved', axis=1)
    y = df['Approved']
    
    # Handling categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Label encoding for categorical
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Save the preprocessing objects for inference
    os.makedirs('model', exist_ok=True)
    joblib.dump(label_encoders, 'model/label_encoders.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, numerical_cols

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, num_cols = load_and_preprocess_data()
    print("Preprocessing completed.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
