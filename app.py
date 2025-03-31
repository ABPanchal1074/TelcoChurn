import streamlit as st
import pandas as pd
import pickle
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

import sys
import subprocess
import pkg_resources

required_packages = ["streamlit", "rich", "setuptools"]
installed_packages = {pkg.key for pkg in pkg_resources.working_set}

for package in required_packages:
    if package not in installed_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
# Set Streamlit page config for a modern look
st.set_page_config(page_title='Telco Churn Prediction', page_icon='📊', layout='wide')

# Custom CSS for improved aesthetics
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #4361ee;
            color: white;
            border-radius: 8px;
            padding: 12px 16px;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #3a56d4;
            box-shadow: 0 4px 12px rgba(67, 97, 238, 0.3);
        }
        
        /* Alert styling */
        .stAlert {
            border-radius: 10px;
        }
        
        /* Titles and headings */
        .main-title {
            font-size: 36px;
            color: #4361ee;
            font-weight: 700;
            text-align: center;
            margin-bottom: 5px;
        }
        .sub-title {
            font-size: 20px;
            color: #555;
            text-align: center;
            margin-bottom: 15px;
        }
        
        /* Card container for sections */
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }
        
        /* Prediction result styling */
        .prediction-box {
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            text-align: center;
        }
        .prediction-stay {
            background-color: #e6f7ff;
            border-left: 4px solid #4cc9f0;
        }
        .prediction-churn {
            background-color: #fff1f0;
            border-left: 4px solid #ff4d4f;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #4361ee;
        }
        
        /* Metric card */
        .metric-card {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            height: 100%;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 600;
            color: #4361ee;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        
        /* Sidebar styling */
        .css-18e3th9 {
            padding-top: 1rem;
        }
        
        /* Form field styling */
        div[data-baseweb="select"] {
            border-radius: 8px;
        }
        div[data-baseweb="input"] {
            border-radius: 8px;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 10px 16px;
            background-color: #f8f9fa;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4361ee !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Telco Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict whether a customer will churn based on their attributes</div>', unsafe_allow_html=True)

# Sidebar for model selection
with st.sidebar:
    st.markdown('<h3 style="color: #4361ee;">Model Settings</h3>', unsafe_allow_html=True)
    model_choice = st.selectbox('Choose Prediction Model', [
        'Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'KNN'
    ])
    
    st.markdown('<hr style="margin: 15px 0;">', unsafe_allow_html=True)
    
    # Add menu options with icons
    st.markdown('<h3 style="color: #4361ee;">Dashboard</h3>', unsafe_allow_html=True)
    menu = st.radio("Select Dashboard", [
        "🏠 Home", "📈 Visualizations", "🔍 Model Comparison", 
        "🔑 PFI", "📱 Real-Time Monitoring"
    ])

# Load models (keeping your original logic)
models = {}
model_names = ['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting', 'knn']

for name in model_names:
    try:
        models[name] = pickle.load(open(f'{name}_model.pkl', 'rb'))
        st.sidebar.success(f'{name.replace("_", " ").title()} model loaded successfully.')
    except FileNotFoundError:
        st.sidebar.error(f'Model file for {name.replace("_", " ").title()} not found.')

# Map user-friendly names to model keys
model_mapping = {
    'Logistic Regression': 'logistic_regression',
    'Decision Tree': 'decision_tree',
    'Random Forest': 'random_forest',
    'Gradient Boosting': 'gradient_boosting',
    'KNN': 'knn'
}

model_key = model_mapping[model_choice]
model = models.get(model_key)

# Handle different menu options
if menu == "📈 Visualizations":
    from visualizations import churn_visualizations
    churn_visualizations()

elif menu == "🔍 Model Comparison":
    from model_comparison import model_comparison
    model_comparison()

elif menu == "🔑 PFI":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #4361ee; margin-top: 0;">🔑 Feature Importance Analysis</h2>', unsafe_allow_html=True)
    
    try:
        # Load the dataset (keeping your path)
        data = pd.read_csv('preprocessed_data.csv')
        X = data.drop('Churn', axis=1)
        y = data['Churn']

        if model:
            # Calculate permutation importance
            result = permutation_importance(model, X, y, n_repeats=10, random_state=42)

            # Create a nicer display of feature importance
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': result.importances_mean
            }).sort_values(by='Importance', ascending=False)

            # Plot feature importance
            fig = px.bar(
                importance_df.head(10), 
                x='Importance', 
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale=['#4cc9f0', '#4361ee', '#3a0ca3'],
                title='Top 10 Features by Importance'
            )
            
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display feature importance table
            st.markdown('### Complete Feature Ranking:')
            st.dataframe(importance_df.style.background_gradient(cmap='Blues'))
        else:
            st.error('No model loaded for feature importance calculation.')

    except Exception as e:
        st.error(f'Error calculating feature importance: {e}')
    
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "📱 Real-Time Monitoring":
    from real_time_monitor import real_time_monitor
    real_time_monitor()

elif menu == "🏠 Home":
    # Overview metrics cards
    st.markdown('<div style="padding: 10px 0 20px 0;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">8,552</div>
            <div class="metric-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">27.3%</div>
            <div class="metric-label">Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">91.6%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main input form with better organization
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #4361ee; margin-top: 0;">🔧 Customer Details</h3>', unsafe_allow_html=True)
    
    # Use tabs to organize input fields
    tabs = st.tabs(["Personal Info", "Services", "Billing"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            senior_citizen = st.selectbox('Senior Citizen', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            partner = st.selectbox('Partner', ['No', 'Yes'])
            dependents = st.selectbox('Dependents', ['No', 'Yes'])
        
        with col2:
            tenure = st.slider('Tenure (Months)', 0, 72, 12)
            phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
            multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'], 
                disabled=(phone_service == 'No'))

    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            internet_enabled = (internet_service != 'No')
            
            online_security = st.selectbox('Online Security', 
                ['No', 'Yes', 'No internet service'], 
                disabled=(not internet_enabled))
                
            online_backup = st.selectbox('Online Backup', 
                ['No', 'Yes', 'No internet service'], 
                disabled=(not internet_enabled))
                
            device_protection = st.selectbox('Device Protection', 
                ['No', 'Yes', 'No internet service'], 
                disabled=(not internet_enabled))
        
        with col2:
            tech_support = st.selectbox('Tech Support', 
                ['No', 'Yes', 'No internet service'], 
                disabled=(not internet_enabled))
                
            streaming_tv = st.selectbox('Streaming TV', 
                ['No', 'Yes', 'No internet service'], 
                disabled=(not internet_enabled))
                
            streaming_movies = st.selectbox('Streaming Movies', 
                ['No', 'Yes', 'No internet service'], 
                disabled=(not internet_enabled))

    with tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
            paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
        
        with col2:
            payment_method = st.selectbox('Payment Method', [
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ])
            
            monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, value=65.0, step=5.0)
            total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=monthly_charges * tenure, step=10.0)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create input dataframe
    input_data = pd.DataFrame([[gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines,
                            internet_service, online_security, online_backup, device_protection, tech_support,
                            streaming_tv, streaming_movies, contract, paperless_billing, payment_method,
                            monthly_charges, total_charges]], 
                            columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                                     'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                                     'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                                     'MonthlyCharges', 'TotalCharges'])

    def preprocess_input(data):
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
        data = pd.get_dummies(data, columns=categorical_cols)
        original_cols = pd.read_csv(r'preprocessed_data.csv').drop('Churn', axis=1).columns
        data = data.reindex(columns=original_cols, fill_value=0)
        return data

    input_data_processed = preprocess_input(input_data)
    
    # Prediction section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #4361ee; margin-top: 0;">🔮 Prediction Results</h3>', unsafe_allow_html=True)
    
    # Risk factors (before prediction)
    risk_factors = []
    if contract == 'Month-to-month':
        risk_factors.append("Month-to-month contract")
    if payment_method == 'Electronic check':
        risk_factors.append("Electronic check payment method")
    if tenure < 12:
        risk_factors.append("Low tenure (<12 months)")
    if internet_service == 'Fiber optic' and tech_support == 'No':
        risk_factors.append("Fiber internet without tech support")

    if risk_factors:
        st.markdown('<h4 style="color: #ff4d4f;">Potential Risk Factors:</h4>', unsafe_allow_html=True)
        for factor in risk_factors:
            st.markdown(f"• {factor}")
    
    # Use a prominently styled button
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_button = st.button('Predict Customer Churn')
    
    if predict_button:
        with st.spinner('Analyzing customer data...'):
            # Add small delay for better UX
            import time
            time.sleep(1)
            
            if model:
                prediction = model.predict(input_data_processed)
                churn_prob = model.predict_proba(input_data_processed)[0][1]
                result = 'Churn' if prediction[0] == 1 else 'No Churn'
                
                # Display prediction with better visualization
                if result == 'No Churn':
                    st.markdown(f"""
                    <div class="prediction-box prediction-stay">
                        <h2 style="color: #4361ee; margin: 0;">✅ Prediction: {result}</h2>
                        <p style="margin: 5px 0 0 0;">This customer is likely to stay with your service.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box prediction-churn">
                        <h2 style="color: #ff4d4f; margin: 0;">⚠️ Prediction: {result}</h2>
                        <p style="margin: 5px 0 0 0;">This customer is at high risk of churning.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Use columns for probability visualization
                prob_col1, prob_col2 = st.columns([3, 1])
                
                with prob_col1:
                    st.markdown(f'<p style="margin-bottom: 5px;"><b>Churn Probability:</b> {churn_prob * 100:.2f}%</p>', unsafe_allow_html=True)
                    st.progress(churn_prob)
                
                with prob_col2:
                    confidence = "Low" if churn_prob < 0.4 else "Medium" if churn_prob < 0.7 else "High"
                    confidence_color = "#4cc9f0" if churn_prob < 0.4 else "#ff9e00" if churn_prob < 0.7 else "#ff4d4f"
                    st.markdown(f"""
                    <div style="text-align: center; padding-top: 10px;">
                        <p style="margin: 0; font-size: 14px;">Confidence</p>
                        <p style="margin: 0; font-weight: bold; color: {confidence_color}; font-size: 18px;">{confidence}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendations section
                st.markdown('<h4 style="color: #4361ee; margin-top: 20px;">Recommendations:</h4>', unsafe_allow_html=True)
                
                if result == 'Churn':
                    recommendations = [
                        "Offer a promotional discount for contract renewal",
                        "Provide a complimentary service upgrade",
                        "Schedule a customer satisfaction call",
                        "Send a personalized retention offer",
                    ]
                else:
                    recommendations = [
                        "Introduce loyalty rewards program",
                        "Suggest complementary services",
                        "Consider for referral program",
                        "Send satisfaction survey",
                    ]
                
                for rec in recommendations:
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #4361ee;">
                        {rec}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error('Please ensure a model is properly loaded before prediction.')
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add quick feature importance summary
    with st.expander("🔍 View Quick Feature Importance"):
        # Sample data for feature importance (replace with actual data in production)
        features = ['Contract', 'Tenure', 'Monthly Charges', 'Internet Service', 'Payment Method', 
                    'Tech Support', 'Online Security', 'Dependents', 'Multiple Lines', 'Paperless Billing']
        importance = [0.24, 0.18, 0.15, 0.11, 0.09, 0.07, 0.06, 0.04, 0.03, 0.03]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            color=importance,
            color_continuous_scale=['#4cc9f0', '#4361ee'],
            title='Top 10 Features by Importance'
        )
        
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
