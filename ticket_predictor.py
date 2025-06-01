# ticket_predictor.py

import joblib
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from entity_extraction import extract_entities

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load models
issue_model = joblib.load("models/issue_type_model.pkl")
urgency_model = joblib.load("models/urgency_level_model.pkl")

# Load TF-IDF vectorizer
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Sentiment scoring
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

def predict_ticket(ticket_text):
    clean_text = preprocess_text(ticket_text)
    tfidf_feat = tfidf_vectorizer.transform([clean_text])
    
    ticket_length = len(ticket_text)
    sentiment_score = sia.polarity_scores(ticket_text)['compound']
    
    additional_feats = pd.DataFrame([[ticket_length, sentiment_score]], columns=["ticket_length", "sentiment_score"])
    full_input = np.hstack([tfidf_feat.toarray(), additional_feats.values])

    issue_pred = issue_model.predict(full_input)[0]
    urgency_pred = urgency_model.predict(full_input)[0]
    entities = extract_entities(ticket_text)

    return {
        "issue_type": issue_pred,
        "urgency_level": urgency_pred,
        "entities": entities
    }

if __name__ == "__main__":
    sample_ticket = "Order #12345 for FitRun Treadmill is late. Also, the EcoBreeze AC has stopped working."
    result = predict_ticket(sample_ticket)
    print("\nPrediction Result:")
    print(result)
    print()
