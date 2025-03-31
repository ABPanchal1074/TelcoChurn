import streamlit as st
import pandas as pd
import plotly.express as px

def churn_visualizations():
    st.title("📈 Interactive Data Visualizations")
    data = pd.read_csv(r'C:\Users\Aniket\TelcoChurn\preprocessed_data.csv')

    st.subheader("Churn Rate by Gender")
    fig = px.histogram(data, x='gender', color='Churn', barmode='group', title='Churn Rate by Gender')
    st.plotly_chart(fig)

    st.subheader("Churn Rate by Contract Type")
    fig = px.histogram(data, x='Contract', color='Churn', barmode='group', title='Churn Rate by Contract Type')
    st.plotly_chart(fig)

    st.subheader("Monthly Charges Distribution")
    fig = px.box(data, x='Churn', y='MonthlyCharges', title='Monthly Charges Distribution by Churn')
    st.plotly_chart(fig)

if __name__ == "__main__":
    churn_visualizations()
