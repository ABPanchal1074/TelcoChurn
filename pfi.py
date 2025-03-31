import streamlit as st
import pandas as pd
import pickle
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def load_model(name):
    try:
        return pickle.load(open(f'C:\\Users\\Aniket\\TelcoChurn\\{name}_model.pkl', 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def feature_importance_explain():
    st.title("🔍 Feature Importance using Permutation Importance")

    model_name = st.selectbox("Select Model", [
        'logistic_regression', 'decision_tree', 
        'random_forest', 'gradient_boosting', 
        'k-nearest_neighbors'
    ])
    model = load_model(model_name)

    if model is None:
        return

    try:
        data = pd.read_csv(r'C:\Users\Aniket\TelcoChurn\preprocessed_data.csv')
        X = data.drop(['Churn'], axis=1)
        y = data['Churn']
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    try:
        st.write("### Feature Importance")
        
        # Calculate permutation importance
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)

        # Create a DataFrame to display the importance
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False)

        # Plotting the feature importance
        fig, ax = plt.subplots()
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Permutation Feature Importance - {model_name.replace('_', ' ').title()}")
        plt.tight_layout()
        st.pyplot(fig)

        # Display the feature importance table
        st.write("### Feature Importance Table")
        st.dataframe(importance_df)

    except Exception as e:
        st.error(f"Error generating feature importance: {e}")

if __name__ == "__main__":
    feature_importance_explain()
