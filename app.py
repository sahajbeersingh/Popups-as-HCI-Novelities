import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
rf = joblib.load('popup_category_rf_model.joblib')
features = joblib.load('features_encoders.joblib')
target = joblib.load('target_encoder.joblib')

st.set_page_config(page_title="Popup Category Predictor", layout="wide")
st.title("Popup Category Prediction")

feature_descriptions = {
    "Type": "Type of the popup (e.g., Banner, Notification Bar, etc.)",
    "Size": "Size of the popup (e.g., Small, Medium, Large)",
    "Trigger": "What triggers the popup (e.g., Timer, Hover, Scroll)",
    "Design": "Design style of the popup (e.g., Flat, Aggressive)",
    "Has_CTA": "Does the popup have a Call To Action button? (Yes/No)",
    "Position": "Screen position of the popup (e.g., Top Left, Bottom Right)",
    "Content_Type": "Type of content shown (e.g., Discount, Newsletter)"
}

st.sidebar.header("Feature Descriptions and Options")
for feat, desc in feature_descriptions.items():
    options = list(features[feat].classes_)
    st.sidebar.markdown(f"**{feat}**")
    st.sidebar.markdown(f"{desc}")
    st.sidebar.markdown(f"Options: {options}")
    st.sidebar.markdown("---")

st.header("Input Popup Features")

user_input = {}
for feat in features.keys():
    options = list(features[feat].classes_)
    user_input[feat] = st.selectbox(f"Select {feat}", options)

def predict_category(user_input, rf_model, encoders, target_encoder):
    feature_names = list(encoders.keys())
    encoded_input = []
    for feature in feature_names:
        encoded_value = encoders[feature].transform([user_input[feature]])[0]
        encoded_input.append(encoded_value)
    input_df = pd.DataFrame([encoded_input], columns=feature_names)
    predicted_label = rf_model.predict(input_df)[0]
    decoded_output = target_encoder.inverse_transform([predicted_label])[0]
    return decoded_output

if st.button("Predict Category"):
    prediction = predict_category(user_input, rf, features, target)
    st.success(f"✅ Predicted Category: **{prediction}**")
