# Data Preprocessing for Churn Prediction

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the Dataset
dataset = pd.read_csv(r'C:\Users\Aniket\TelcoChurn\Corrected_Telco_Customer_Churn.csv')

# Step 3: Keep Only Necessary Columns
columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
           'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
dataset = dataset[columns]

# Step 4: Handle Missing Values
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset.dropna(inplace=True)

# Step 5: Encoding Categorical Variables
label_enc = LabelEncoder()
for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']:
    dataset[col] = label_enc.fit_transform(dataset[col])

# Step 6: Feature Scaling
scaler = StandardScaler()
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
dataset[numerical_cols] = scaler.fit_transform(dataset[numerical_cols])

# Step 7: Save the Preprocessed Data
dataset.to_csv(r'C:\Users\Aniket\TelcoChurn\preprocessed_data.csv', index=False)
print("Data Preprocessing Completed!")
