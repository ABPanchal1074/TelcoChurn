import streamlit as st
import pandas as pd

def churn_report(prediction, probability):
    st.title("📃 Personalized Churn Report")
    if prediction == 1:
        st.error(f"Customer is likely to churn with a probability of {probability:.2%}.")
    else:
        st.success(f"Customer is not likely to churn with a probability of {1 - probability:.2%}.")
    st.write("#### Key Recommendations")
    st.write("1. Offer personalized discounts or upgrade options.")
    st.write("2. Improve customer service and engagement.")
    st.write("3. Provide loyalty benefits to long-term customers.")
