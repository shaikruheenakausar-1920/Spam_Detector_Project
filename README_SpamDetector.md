# 📧 Email/SMS Spam Detector

## What it does
A web app where you paste any email or text message, and it tells you whether
the message is Spam or Not Spam, along with a confidence percentage.

## How it works
- **Dataset**: spam.csv (~5,500 labeled SMS messages)
- **Model**: Multinomial Naive Bayes (Scikit-learn)
- **Text processing**: CountVectorizer (bag-of-words, English stopwords removed)
- **Accuracy**: 98.4% on test data
- **Frontend**: Streamlit

## How to run
```
cd Spam_Detector_Project
streamlit run app.py
```
This opens automatically in your browser. The model is already trained
(spam_model.pkl + vectorizer.pkl) so it works immediately — no need to retrain.

To retrain from scratch (optional, prints accuracy score):
```
python train_model.py
```
