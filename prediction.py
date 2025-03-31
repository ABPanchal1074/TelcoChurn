# prediction.py

# Prediction

# Step 1: Import Libraries
import pandas as pd
import pickle

# Step 2: Load Preprocessed Data
dataset = pd.read_csv(r'C:\Users\Aniket\TelcoChurn\preprocessed_data.csv')
X = dataset.drop(['Churn'], axis=1)

# Step 3: Load Trained Model (Random Forest as an example)
with open(r'C:\Users\Aniket\TelcoChurn\random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Step 4: Make Predictions
predictions = model.predict(X)

# Step 5: Save Predictions to CSV
output = pd.DataFrame({'Prediction': predictions})
output.to_csv(r'C:\Users\Aniket\TelcoChurn\predictions.csv', index=False)

print("Predictions saved to predictions.csv!")
