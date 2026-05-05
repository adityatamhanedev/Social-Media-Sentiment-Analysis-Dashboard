import pickle
import os
from typing import List, Union
import numpy as np

from src.preprocess import clean_text


class SentimentPredictor:
    """
    Production-style sentiment prediction pipeline.
    Handles:
    - Model loading
    - Vectorizer loading
    - Text preprocessing
    - Batch + single prediction
    """

    def __init__(self, model_path: str, vectorizer_path: str):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

        self.model = None
        self.vectorizer = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model and vectorizer from disk"""

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {self.vectorizer_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        print("✅ Model and Vectorizer loaded successfully!")

    def preprocess(self, texts: Union[str, List[str]]) -> List[str]:
        """
        Clean input text(s)
        """
        if isinstance(texts, str):
            texts = [texts]

        cleaned = [clean_text(text) for text in texts]
        return cleaned

    def transform(self, cleaned_texts: List[str]):
        """
        Convert text to TF-IDF features
        """
        return self.vectorizer.transform(cleaned_texts)

    def predict(self, texts: Union[str, List[str]]):
        """
        Predict sentiment
        """
        cleaned = self.preprocess(texts)
        features = self.transform(cleaned)

        predictions = self.model.predict(features)

        return predictions

    def predict_proba(self, texts: Union[str, List[str]]):
        """
        Predict probabilities (if model supports it)
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model does not support probability prediction")

        cleaned = self.preprocess(texts)
        features = self.transform(cleaned)

        probabilities = self.model.predict_proba(features)

        return probabilities

    def predict_with_confidence(self, texts: Union[str, List[str]]):
        """
        Returns sentiment + confidence score
        """
        preds = self.predict(texts)
        probs = self.predict_proba(texts)

        results = []

        for i in range(len(preds)):
            confidence = np.max(probs[i])
            results.append({
                "text": texts[i] if isinstance(texts, list) else texts,
                "sentiment": preds[i],
                "confidence": round(float(confidence), 3)
            })

        return results