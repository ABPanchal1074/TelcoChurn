# model_building.py

# Model Building and Training

# Step 1: Import Libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Step 2: Load Preprocessed Data
dataset = pd.read_csv(r'C:\Users\Aniket\TelcoChurn\preprocessed_data.csv')

# Step 3: Splitting the Dataset
X = dataset.drop(['Churn'], axis=1)
y = dataset['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize Models
models = {
    'logistic_regression': LogisticRegression(),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'gradient_boosting': GradientBoostingClassifier(),
    'knn': KNeighborsClassifier()
}

# Step 5: Train, Evaluate, and Save Models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name.replace("_", " ").title()} Accuracy: {accuracy * 100:.2f}%')

    # Save the trained model
    with open(f'C:\\Users\\Aniket\\TelcoChurn\\{name}_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print(f'{name.replace("_", " ").title()} model trained and saved successfully!')

print("Model Building and Training Completed!")
