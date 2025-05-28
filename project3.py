import streamlit as st
import pandas as pd
import joblib

# Set page config first
st.set_page_config(page_title="🔮 Sales Forecast", layout="centered")

#Custom CSS for aesthetics
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            background-color: black;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: gray;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: white;
            text-align: center;
            font-size: 2rem;
            margin-top: 20px;
        }
        .stButton button {
            background-color: black;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            transition: background-color 0.3s ease-in-out;
        }
        .stButton button:hover {
            background-color: black;
        }
        .stMetric, .stCaption {
            color: white;
        }
        #logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            max-width: 200px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)



# Load model with caching
@st.cache_resource
def load_model():
    try:
        return joblib.load("prophet.joblib")
    except FileNotFoundError:
        st.error(" Model file not found. Please ensure 'prophet.joblib' is in the app directory.")
        return None

model = load_model()

# 🚀 UI Layout
st.title("📆 Predict Future Sales")

st.markdown("""
Welcome to the **Sales Forecaster** powered by **Prophet**.  
Just pick a date, and we’ll show you the forecasted sales with confidence bounds.
""")

if model:
    selected_date = st.date_input("🗓️ Choose a date to forecast:")

    if selected_date:
        df_future = pd.DataFrame({'ds': [pd.to_datetime(selected_date)]})
        forecast = model.predict(df_future)

        yhat = forecast['yhat'].values[0]
        lower = forecast['yhat_lower'].values[0]
        upper = forecast['yhat_upper'].values[0]

        st.subheader(f"📊 Forecast Results for {selected_date.strftime('%A, %d %B %Y')}")

        col1, col2, col3 = st.columns(3)
        col1.metric("📈 Predicted Sales", f"{yhat:.2f}")
        col2.metric("🔻 Lower Bound", f"{lower:.2f}")
        col3.metric("🔺 Upper Bound", f"{upper:.2f}")

        st.markdown("---")
        st.caption("The predicted values come from Facebook Prophet's time series model. "
                   "The confidence bounds show uncertainty based on the trend and seasonality.")

else:
    st.warning("Model not loaded. Please check your `prophet.joblib` file.")
