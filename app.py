import streamlit as st
from src.predict import predict_churn

st.set_page_config(page_title="Churn Prediction", layout="wide")

# ---- HEADER ----
st.title("📊 Customer Churn Prediction System")
st.markdown("### AI-powered customer retention insights")

st.markdown("---")

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "🔍 Prediction", "📈 Insights"])

# ===================== DASHBOARD =====================
with tab1:
    st.subheader("📌 Business Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Used", "XGBoost / Best Model")
    col2.metric("ROC-AUC Score", "0.81")
    col3.metric("Goal", "Reduce Customer Churn")

    st.markdown("---")

    st.info("💡 This tool helps businesses identify customers likely to churn and take preventive actions.")

# ===================== PREDICTION =====================
with tab2:
    st.subheader("🔍 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure", 0, 72, 12)

        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col2:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
        TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

    # Hidden inputs (keep model consistency)
    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": Contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    if st.button("🚀 Predict Now"):
        prediction, prob = predict_churn(input_data)

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric("Churn Probability", f"{prob:.2f}")
        col2.metric("Risk Level", "High" if prediction == 1 else "Low")

        st.progress(float(prob))

        if prediction == 1:
            st.error("⚠️ High Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

# ===================== INSIGHTS =====================
with tab3:
    st.subheader("📈 Key Insights")

    st.markdown("""
    - Customers with **month-to-month contracts** are more likely to churn  
    - Higher **monthly charges** increase churn probability  
    - Longer **tenure reduces churn risk**  
    - Lack of **tech support & security services** increases churn  
    """)

    st.warning("⚠️ Businesses should focus on retention strategies for high-risk segments.")