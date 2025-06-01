# entity_extraction.py

import re
import pandas as pd
from nltk.tokenize import word_tokenize

PRODUCT_LIST = [
    "SmartWatch V2", "UltraClean Vacuum", "SoundWave 300", "EcoBreeze AC",
    "PhotoSnap Cam", "Vision LED TV", "RoboChef Blender", "FitRun Treadmill"
]

COMPLAINT_KEYWORDS = ["broken", "late", "error", "defect", "malfunction", "missing", "stopped"]

DATE_PATTERN = r"\b(?:\d{1,2}[\s/-](?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\s/-]\d{2,4})\b|\b(?:\d{1,2}[\s/-]\d{1,2}[\s/-]\d{2,4})\b"


def extract_entities(text):
    entities = {
        "products": [],
        "dates": [],
        "complaint_keywords": []
    }

    for product in PRODUCT_LIST:
        if product.lower() in text.lower():
            entities["products"].append(product)

    dates_found = re.findall(DATE_PATTERN, text)
    entities["dates"] = dates_found

    tokens = word_tokenize(text.lower())
    for word in COMPLAINT_KEYWORDS:
        if word in tokens:
            entities["complaint_keywords"].append(word)

    return entities

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_tickets.csv")
    df['entities'] = df['ticket_text'].apply(extract_entities)
    df[['ticket_id', 'entities']].to_json("data/extracted_entities.json", orient="records", lines=True)
    print("Entity extraction complete. Output saved to data/extracted_entities.json")
