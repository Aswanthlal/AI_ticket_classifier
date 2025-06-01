import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# load model
nlp= spacy.load('en_core_web_sm')
stop_words=set(stopwords.words('english'))
# filepath=

def load_data(filepath):
    df=pd.read_excel(filepath)
    return df

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    text=re.sub(r'^a-zA-Z0-9\s','',text.lower())
    doc=nlp(text)
    tokens=[token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

def handle_missing_data(df):
    df['urgency_level']=df['urgency_level'].fillna('Unknown')
    df['issue_type']=df['issue_type'].fillna('Unknown')
    return df

def clean_dataset(df):
    df['clean_text']=df['ticket_text'].apply(preprocess_text)
    return df

def save_cleaned_dataset(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_path= "ai_dev_assignment_tickets_complex_1000.xlsx"
    output_path = "data/cleaned_tickets.csv"

    df=load_data(input_path)
    df=handle_missing_data(df)
    df=clean_dataset(df)
    save_cleaned_dataset(df, output_path)
    print('Data cleaning complete. File saved to', output_path)
    
