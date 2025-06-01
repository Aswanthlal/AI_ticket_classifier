# feature_engineering.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# TF-IDF vectorizer with max features
vectorizer = TfidfVectorizer(max_features=3000)

def add_ticket_length(df):
    df['ticket_length'] = df['clean_text'].apply(lambda x: len(x.split()))
    return df

def add_sentiment_score(df):
    df['sentiment'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

def extract_tfidf_features(df):
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.index = df.index
    return tfidf_df

def combine_features(df, tfidf_df):
    combined_df = pd.concat([df[['ticket_id', 'ticket_length', 'sentiment', 'issue_type', 'urgency_level']], tfidf_df], axis=1)
    return combined_df

if __name__ == "__main__":
    input_path = "data/cleaned_tickets.csv"
    df = pd.read_csv(input_path)

    df = add_ticket_length(df)
    df = add_sentiment_score(df)
    tfidf_df = extract_tfidf_features(df)
    final_df = combine_features(df, tfidf_df)

    final_df.to_csv("data/feature_engineered.csv", index=False)
    print("Feature engineering complete. Output saved to data/feature_engineered.csv")
