# AI-Powered Credit Card Approval Prediction System

## Problem Statement
Banks and financial institutions receive thousands of credit card applications every day. A significant portion of these are rejected due to factors such as high existing loan balances, insufficient income levels, or excessive credit inquiries. Manually reviewing each application is both time-consuming and highly prone to human error, making it an inefficient process at scale.

## Overview
This project automates the credit card approval decision using machine learning. By training a predictive model on historical applicant data, the system evaluates financial and demographic inputs to determine whether an applicant is likely to be approved or rejected just as real banks do. 

Four classification algorithms are applied:
- Logistic Regression
- Random Forest
- XGBoost (Gradient Boosting)
- Decision Tree

The best-performing model is saved and integrated into a Flask web application. The project also includes an IBM Watson Machine Learning deployment pipeline, enabling the solution to be hosted on the cloud for scalable, real-time credit card approval predictions accessible through an intuitive user interface.

## Use Cases

### Scenario 1: Automated Credit Card Application Screening
A bank credit analyst enters a new applicant’s financial profile including income type, employment duration, and credit history into the web application. The model returns an instant approval or rejection prediction, enabling the analyst to prioritize applications that require further review.

### Scenario 2: High-Risk Applicant Identification and Compliance Review
A financial compliance officer uses the platform to batch-screen applicants with past-due loan records. The feature engineering pipeline converts multi-class payment status codes into binary labels, allowing the model to clearly classify high-risk applicants as ineligible for a new credit card.

### Scenario 4: Customer Self-Service Credit Card Eligibility Check
A prospective customer uses the web application to enter personal and financial details such as income level, employment status, and credit history before submitting a formal credit card application. The system instantly predicts the likelihood of approval, helping the customer understand their eligibility and reducing unnecessary application rejections.

## How to Run Locally
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Generate synthetic data:** `python generate_data.py`
3. **Preprocess data and train model:** `python preprocess.py` then `python train_model.py`
4. **Run the web application:** `python app.py`
5. Open browser at `http://127.0.0.1:5000`
