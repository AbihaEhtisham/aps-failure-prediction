# src/app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Scania APS Predictor",
    page_icon="🚛",
    layout="wide"
)

st.title("🚛 Scania APS Failure Prediction")
st.markdown("### Enterprise Predictive Maintenance Dashboard")

# Load model artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/best_model.pkl')
        threshold = joblib.load('models/optimal_threshold.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, threshold, feature_names
    except:
        return None, 0.3, []

model, threshold, feature_names = load_artifacts()

if model is None:
    st.warning("⚠️ Model files not found! Please make sure models are in the 'models/' folder.")
    st.info("Run the notebooks first to generate model files.")
else:
    st.success(f"✅ Model loaded! Optimal threshold: {threshold:.3f}")

# Sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **Business Cost Matrix**
    - FP Cost: 10
    - FN Cost: 500
    - Optimal Threshold: {threshold:.3f}
    """
)

# Prediction section
st.subheader("🔮 Make a Prediction")

# Create input fields
if model is not None and feature_names:
    cols = st.columns(3)
    input_values = {}
    
    # Show first 15 features
    for i, feature in enumerate(feature_names[:15]):
        col_idx = i % 3
        with cols[col_idx]:
            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.2f"
            )
    
    if st.button("🚀 Predict", type="primary"):
        try:
            # Make prediction
            input_df = pd.DataFrame([input_values])
            y_proba = model.predict_proba(input_df)[0][1]
            y_pred = int(y_proba >= threshold)
            
            # Show results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                prediction_text = "APS Failure" if y_pred == 1 else "Other Failure"
                color = "#DC3545" if y_pred == 1 else "#28A745"
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid {color};">
                    <h3>🎯 Prediction</h3>
                    <h2 style="color: {color};">{prediction_text}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
                    <h3>📊 Probability</h3>
                    <h2>{y_proba:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                confidence = abs(y_proba - 0.5) * 2
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
                    <h3>🔒 Confidence</h3>
                    <h2>{confidence:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk gauge
            st.subheader("📊 Risk Assessment")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=y_proba * 100,
                title={'text': "APS Failure Risk"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#DC3545" if y_proba > 0.5 else "#28A745"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Built for Scania APS Failure Prediction DataDive 2026")