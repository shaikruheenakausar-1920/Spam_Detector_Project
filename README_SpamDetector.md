# 📧 Email/SMS Spam Detector

A machine learning web app that classifies text messages as Spam or Not Spam, built with Python, Scikit-learn, and Streamlit.

## 🔗 Demo
> Add a screenshot or GIF of the app here after running it locally.
> Example: `![App Screenshot](screenshot.png)`

## 📌 Problem Statement
Spam messages (junk mail, scams, phishing) are a common nuisance. This project builds a text classification model that automatically detects whether a message is spam or legitimate (ham).

## 🛠 Tech Stack
- Python
- Pandas (data handling)
- Scikit-learn (machine learning + text vectorization)
- Streamlit (web app)

## 📊 Dataset
`spam.csv` — ~5,500 labeled SMS messages, with columns:
`v1` (label: ham/spam), `v2` (message text)

## 🧠 Approach
1. **Preprocessing**: Kept only the label and message columns; converted labels to 0 (ham) and 1 (spam).
2. **Text Vectorization**: Used `CountVectorizer` (bag-of-words model) with English stopwords removed, to convert text messages into numeric features.
3. **Model**: Trained a `Multinomial Naive Bayes` classifier — a standard, efficient choice for text classification.
4. **Evaluation**: Split data 80/20 (train/test) and evaluated using accuracy.

## ✅ Result
**Accuracy: 98.4%** on the test set.

## 📥 Sample Prediction
| Message | Prediction |
|---|---|
| "Congratulations! You won a free prize, click here to claim!" | 🚨 Spam |
| "Hey, are we still meeting for lunch today?" | ✅ Not Spam |

## 🚀 How to Run
```bash
pip install streamlit pandas scikit-learn
streamlit run app.py
```
The app opens in your browser. The model (`spam_model.pkl` + `vectorizer.pkl`) is already trained, so it works immediately.

To retrain the model from scratch:
```bash
python train_model.py
```

## 📂 Project Structure
```
Spam_Detector_Project/
├── app.py              # Streamlit app
├── train_model.py       # Model training script
├── spam.csv              # Dataset
├── spam_model.pkl        # Trained model
└── vectorizer.pkl        # Text vectorizer
```

## 💡 What I Learned
- Converting raw text into numeric features using CountVectorizer
- Using Naive Bayes for text classification
- Evaluating classification models using accuracy and confidence scores
- Deploying a trained NLP model into an interactive Streamlit app
