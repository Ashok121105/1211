import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_records=5000):
    np.random.seed(42)
    
    # 1. Gender
    gender = np.random.choice(['Male', 'Female'], num_records, p=[0.55, 0.45])
    
    # 2. Age (18 to 70)
    age = np.random.randint(18, 70, num_records)
    
    # 3. Income Type
    income_types = ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student']
    income_type = np.random.choice(income_types, num_records, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    
    # 4. Employment Status
    emp_status = np.random.choice(['Employed', 'Unemployed', 'Self-Employed'], num_records, p=[0.7, 0.1, 0.2])
    
    # 5. Annual Income
    annual_income = np.random.normal(60000, 25000, num_records)
    annual_income = np.clip(annual_income, 15000, 250000) # clip to realistic bounds
    
    # 6. Education
    education_levels = ['Secondary', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']
    education = np.random.choice(education_levels, num_records, p=[0.4, 0.4, 0.1, 0.08, 0.02])
    
    # 7. Marital Status
    marital_status = np.random.choice(['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'], num_records, p=[0.5, 0.3, 0.1, 0.05, 0.05])
    
    # 8. Housing Type
    housing = np.random.choice(['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment'], num_records, p=[0.8, 0.05, 0.05, 0.05, 0.025, 0.025])
    
    # 9. Number of Children
    children = np.random.poisson(lam=1, size=num_records)
    children = np.clip(children, 0, 5)
    
    # 10. Employment Duration (Years)
    emp_duration = np.random.randint(0, 40, num_records)
    emp_duration = np.where(emp_status == 'Unemployed', 0, emp_duration)
    
    # 11. Existing Loan Balance
    loan_balance = np.random.exponential(scale=15000, size=num_records)
    
    # 12. Credit History (Years)
    credit_history = np.random.randint(0, 20, num_records)
    
    # 13. Payment Status (1 = Good, 0 = Bad/Missed Payments)
    payment_status = np.random.choice([1, 0], num_records, p=[0.8, 0.2])
    
    # 14. Credit Inquiries (Last 6 months)
    credit_inquiries = np.random.poisson(lam=0.5, size=num_records)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Gender': gender,
        'Age': age,
        'Income_Type': income_type,
        'Employment_Status': emp_status,
        'Annual_Income': annual_income,
        'Education': education,
        'Marital_Status': marital_status,
        'Housing_Type': housing,
        'Num_Children': children,
        'Employment_Duration': emp_duration,
        'Existing_Loan_Balance': loan_balance,
        'Credit_History_Length': credit_history,
        'Payment_Status': payment_status,
        'Credit_Inquiries': credit_inquiries
    })
    
    # 15. Target Variable: Approved (1) or Rejected (0)
    # Let's create some logical rules to generate the target
    score = np.zeros(num_records)
    
    # Positive factors
    score += (data['Annual_Income'] / 10000) * 1.5
    score += data['Credit_History_Length'] * 2
    score += (data['Payment_Status'] == 1) * 20
    score += (data['Employment_Status'] == 'Employed') * 10
    
    # Negative factors
    score -= data['Credit_Inquiries'] * 5
    score -= (data['Existing_Loan_Balance'] / 10000) * 2
    score -= (data['Payment_Status'] == 0) * 30
    
    # Add some noise
    score += np.random.normal(0, 10, num_records)
    
    # Threshold for approval (approx 60% approval rate)
    threshold = np.percentile(score, 40)
    data['Approved'] = (score >= threshold).astype(int)
    
    return data

if __name__ == "__main__":
    os.makedirs('dataset', exist_ok=True)
    df = generate_synthetic_data(10000)
    output_path = os.path.join('dataset', 'credit_card_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset generated and saved to {output_path}")
    print(df.head())
    print("\nTarget Distribution:")
    print(df['Approved'].value_counts(normalize=True))
