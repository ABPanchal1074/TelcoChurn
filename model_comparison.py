import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_model(name):
    return pickle.load(open(f'C:\\Users\\Aniket\\TelcoChurn\\{name}_model.pkl', 'rb'))

def model_comparison():
    st.title("📊 Model Comparison Dashboard")

    model_names = ['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting', 'k-nearest_neighbors']
    data = pd.read_csv(r'C:\Users\Aniket\TelcoChurn\preprocessed_data.csv')

    X = data.drop(['Churn'], axis=1)
    y = data['Churn']

    results = {}
    for name in model_names:
        model = load_model(name)
        accuracy = accuracy_score(y, model.predict(X))
        results[name] = accuracy

    st.write("### Model Accuracy Comparison")
    st.write(pd.DataFrame(results.items(), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False))

if __name__ == "__main__":
    model_comparison()
