import streamlit as st
import numpy as np
import joblib

try:
    model = joblib.load('salary_predictor_model.pkl') 
except Exception as e:
    st.error(f"Error loading model: {e}")

st.markdown("""
    <style>
        .title {
            color: #f4d034;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.markdown('<h1 class="title">Salary Predictor</h1>', unsafe_allow_html=True)

years = st.number_input("Years of Experience", min_value=0, value=0)

if st.button("Predict Salary"):
    input_data = np.array([[years]])  
    
    prediction = model.predict(input_data)
    
    st.write(f"Predicted Salary: ₹{prediction[0]:,.2f}")  


footer = """
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0);
        text-align: center;
        padding: 10px;
        color: white;
        opacity: 0.5;
    }

    a{
        text-decoration: none;
        underline: none;
    }

    a:hover{
        color: #f4d03f;
        text-decoration: none;
        underline: none;
    }
</style>
<div class="footer">
    Made with 💖 by <a href="https://github.com/devarsheecodess"> @devarsheecodess </a>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)