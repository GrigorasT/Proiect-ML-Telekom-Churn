import numpy as np
import pandas as pd

def load_telecom_data(n_samples=7043, random_state=42):
    """
    PART 1: ÎNCARCĂ DATE (Dataset sintetic Telecom Churn)
    Generează date brute simulate pentru proiect.
    """
    np.random.seed(random_state)

    tenure          = np.random.randint(0, 72, n_samples)
    monthly_charges = np.random.uniform(18, 118, n_samples)
    total_charges   = tenure * monthly_charges + np.random.normal(0, 50, n_samples)
    total_charges   = np.clip(total_charges, 0, None)

    contract_type    = np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                         n_samples, p=[0.55, 0.25, 0.20])
    payment_method   = np.random.choice(['Electronic check', 'Mailed check',
                                          'Bank transfer', 'Credit card'],
                                         n_samples, p=[0.34, 0.23, 0.22, 0.21])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'],
                                         n_samples, p=[0.34, 0.44, 0.22])
    tech_support     = np.random.choice(['Yes', 'No', 'No internet service'],
                                         n_samples, p=[0.29, 0.49, 0.22])
    online_security  = np.random.choice(['Yes', 'No', 'No internet service'],
                                         n_samples, p=[0.28, 0.50, 0.22])
    senior_citizen    = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    paperless_billing = np.random.choice([0, 1], n_samples, p=[0.41, 0.59])

    #Probabilitate churn realistă
    churn_prob = (
        0.35 * (contract_type == 'Month-to-month').astype(float)
        + 0.15 * (internet_service == 'Fiber optic').astype(float)
        - 0.20 * (tech_support == 'Yes').astype(float)
        - 0.001 * tenure
        + 0.002 * monthly_charges
        + 0.05 * senior_citizen
        + np.random.uniform(-0.1, 0.1, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_prob * 3))
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract_type,
        'PaymentMethod': payment_method,
        'InternetService': internet_service,
        'TechSupport': tech_support,
        'OnlineSecurity': online_security,
        'SeniorCitizen': senior_citizen,
        'PaperlessBilling': paperless_billing,
        'Churn': churn
    })
    
    return df