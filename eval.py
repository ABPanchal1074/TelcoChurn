# model_evaluation.py

# Model Evaluation

# Step 1: Import Libraries
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Preprocessed Data
dataset = pd.read_csv(r'C:\Users\Aniket\TelcoChurn\preprocessed_data.csv')
X = dataset.drop(['Churn'], axis=1)
y = dataset['Churn']

# Step 3: Load Trained Models
models = ['logistic_regression_model.pkl', 'decision_tree_model.pkl', 'random_forest_model.pkl', 'gradient_boosting_model.pkl', 'k-nearest_neighbors_model.pkl']

# Step 4: Evaluate Models
for model_file in models:
    with open(f'C:\\Users\\Aniket\\TelcoChurn\\{model_file}', 'rb') as file:
        model = pickle.load(file)
    y_pred = model.predict(X)
    print(f'Performance of {model_file} :')
    print('Accuracy:', accuracy_score(y, y_pred))
    print('Classification Report:\n', classification_report(y, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y, y_pred))
    print('\n' + '-'*50 + '\n')

print("Model Evaluation Completed!")
