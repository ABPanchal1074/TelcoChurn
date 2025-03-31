import streamlit as st
import pandas as pd

def real_time_monitor():
    st.title("⏱️ Real-Time Churn Monitoring")
    data = pd.read_csv(r'C:\Users\Aniket\TelcoChurn\preprocessed_data.csv')
    st.dataframe(data.tail(10))

