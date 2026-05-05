import pandas as pd
from src.preprocess import clean_text
from src.feature_engineering import get_vectorizer
from src.model import train_model, save_model
import pickle
from src.data_loader import load_data, preview_data, check_nulls, validate_columns

file_path = "data/social_media_data.csv"

df = load_data(file_path)

validate_columns(df, ["text", "sentiment"])
preview_data(df)
check_nulls(df)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

df = pd.read_csv('data/social_media_data.csv')

df['cleaned'] = df['text'].apply(clean_text)

vectorizer = get_vectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['sentiment']

model = train_model(X, y)

save_model(model, 'models/sentiment_model.pkl')

print("Model trained successfully!")