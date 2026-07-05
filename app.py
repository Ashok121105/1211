from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import os
import traceback
import sqlite3
import json

app = Flask(__name__)
app.secret_key = 'super_secret_credit_guard_key_2026'

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            input_data TEXT,
            prediction TEXT,
            confidence TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Load model and preprocessors
try:
    model = joblib.load('model/model.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')
    scaler = joblib.load('model/scaler.pkl')
except Exception as e:
    print(f"Error loading models. Make sure you have trained the model first. {e}")
    model, label_encoders, scaler = None, None, None

@app.route('/')
def home():
    # Fetch statistics for the dashboard
    total_approved = 0
    total_rejected = 0
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute("SELECT prediction, COUNT(*) FROM requests GROUP BY prediction")
        results = c.fetchall()
        for row in results:
            if row[0] == "Approved":
                total_approved = row[1]
            elif row[0] == "Rejected":
                total_rejected = row[1]
        conn.close()
    except Exception as e:
        print(f"Error fetching stats: {e}")
        
    return render_template('index.html', approved=total_approved, rejected=total_rejected)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True
            return redirect(url_for('admin'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('home'))

@app.route('/admin')
def admin():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    # Fetch all prediction logs from the SQLite database
    logs = []
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute("SELECT id, timestamp, prediction, confidence, input_data FROM requests ORDER BY timestamp DESC LIMIT 50")
        logs = c.fetchall()
        conn.close()
    except Exception as e:
        print(f"Error fetching logs: {e}")
        
    return render_template('admin.html', logs=logs)

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not label_encoders or not scaler:
        return render_template('result.html', prediction_text="Model is not loaded properly. Please train the model first.")
        
    try:
        # Get data from form
        form_data = request.form.to_dict()
        
        # Create DataFrame
        input_data = pd.DataFrame([form_data])
        
        # Numeric conversions
        numeric_fields = ['Age', 'Annual_Income', 'Num_Children', 'Employment_Duration', 
                          'Existing_Loan_Balance', 'Credit_History_Length', 'Payment_Status', 'Credit_Inquiries']
                          
        for col in numeric_fields:
            if col in input_data.columns:
                input_data[col] = pd.to_numeric(input_data[col])
                
        # Categorical processing
        for col, le in label_encoders.items():
            if col in input_data.columns:
                # Handle unknown categories gracefully
                try:
                    input_data[col] = le.transform(input_data[col].astype(str))
                except ValueError:
                    # If unseen category, use the most frequent one (mode) from training data 
                    # For simplicity, fallback to the first class in le.classes_
                    input_data[col] = 0 
                    
        # Define the exact column order expected by the model
        expected_cols = ['Gender', 'Age', 'Income_Type', 'Employment_Status',
       'Annual_Income', 'Education', 'Marital_Status', 'Housing_Type',
       'Num_Children', 'Employment_Duration', 'Existing_Loan_Balance',
       'Credit_History_Length', 'Payment_Status', 'Credit_Inquiries']
       
        # Ensure correct order and missing columns handled
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0
                
        input_data = input_data[expected_cols]

        # Scaling
        numeric_cols_in_data = input_data.select_dtypes(include=['int64', 'float64']).columns
        # Only scale the exact columns the scaler was fit on
        scaler_cols = scaler.feature_names_in_
        input_data[scaler_cols] = scaler.transform(input_data[scaler_cols])

        # Prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] if hasattr(model, 'predict_proba') else 0.5
        
        if prediction[0] == 1:
            result = "Approved"
            confidence = f"{probability * 100:.2f}%"
        else:
            result = "Rejected"
            confidence = f"{(1 - probability) * 100:.2f}%"

        # Log to Database
        try:
            conn = sqlite3.connect('predictions.db')
            c = conn.cursor()
            c.execute("INSERT INTO requests (input_data, prediction, confidence) VALUES (?, ?, ?)", 
                     (json.dumps(form_data), result, confidence))
            conn.commit()
            conn.close()
        except Exception as db_e:
            print(f"Database logging error: {db_e}")

        return render_template('result.html', prediction_text=f"Credit Card Application is {result}", confidence=confidence)

    except Exception as e:
        print(traceback.format_exc())
        return render_template('result.html', prediction_text=f"Error occurred during prediction: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not model or not label_encoders or not scaler:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        req_data = request.get_json()
        if not req_data:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        input_data = pd.DataFrame([req_data])
        
        # Numeric conversions
        numeric_fields = ['Age', 'Annual_Income', 'Num_Children', 'Employment_Duration', 
                          'Existing_Loan_Balance', 'Credit_History_Length', 'Payment_Status', 'Credit_Inquiries']
        for col in numeric_fields:
            if col in input_data.columns:
                input_data[col] = pd.to_numeric(input_data[col])
                
        # Categorical processing
        for col, le in label_encoders.items():
            if col in input_data.columns:
                try:
                    input_data[col] = le.transform(input_data[col].astype(str))
                except ValueError:
                    input_data[col] = 0 
                    
        expected_cols = ['Gender', 'Age', 'Income_Type', 'Employment_Status',
       'Annual_Income', 'Education', 'Marital_Status', 'Housing_Type',
       'Num_Children', 'Employment_Duration', 'Existing_Loan_Balance',
       'Credit_History_Length', 'Payment_Status', 'Credit_Inquiries']
       
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0
                
        input_data = input_data[expected_cols]
        scaler_cols = scaler.feature_names_in_
        input_data[scaler_cols] = scaler.transform(input_data[scaler_cols])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] if hasattr(model, 'predict_proba') else 0.5
        
        result = "Approved" if prediction[0] == 1 else "Rejected"
        confidence_val = float(probability) if prediction[0] == 1 else float(1 - probability)

        return jsonify({
            "prediction": result,
            "confidence_score": confidence_val,
            "probability_approved": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
