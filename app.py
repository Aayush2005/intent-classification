import streamlit as st
from inference import predict_intent

st.set_page_config(page_title="Insurance Intent Classifier", layout="centered")

st.title("🤖 Insurance Intent Classifier")
st.markdown("Enter a customer query and get the predicted **intent**.")

user_input = st.text_area("Enter Query:", height=100)

if st.button("Predict"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            intent = predict_intent(user_input)
        st.success(f"**Predicted Intent:** `{intent}`")
    else:
        st.warning("Please enter a valid input.")
