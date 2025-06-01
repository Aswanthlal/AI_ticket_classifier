# multi_task_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def train_and_evaluate(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n===== {model_name} Report =====")
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, f"models/{model_name.replace(' ', '_').lower()}_model.pkl")
    return clf

if __name__ == "__main__":
    df = pd.read_csv("data/feature_engineered.csv")
    
    # Drop irrelevant columns
    X = df.drop(columns=['ticket_id', 'issue_type', 'urgency_level'])

    # Train issue type classifier
    issue_type_clf = train_and_evaluate(X, df['issue_type'], "Issue Type")

    # Train urgency level classifier
    urgency_clf = train_and_evaluate(X, df['urgency_level'], "Urgency Level")

    print("Models trained and saved to models/ directory.")
