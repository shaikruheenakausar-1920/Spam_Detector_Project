import streamlit as st
import pickle

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Email/SMS Spam Detector")
st.write("Type or paste a message below to check if it's Spam or Not Spam.")

message = st.text_area("Enter your message:", height=150, placeholder="e.g. Congratulations! You won a free prize, click here to claim!")

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]
        probability = model.predict_proba(message_vec)[0]

        if prediction == 1:
            st.error(f"🚨 This message is SPAM! (Confidence: {round(probability[1]*100, 2)}%)")
        else:
            st.success(f"✅ This message is NOT Spam (Confidence: {round(probability[0]*100, 2)}%)")
