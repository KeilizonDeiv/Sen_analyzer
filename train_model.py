"""
Sentiment Analysis Model Trainer
Trains a custom sentiment classifier on IMDB movie reviews dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def train(self, X_train, y_train):
        """Train the sentiment analysis model"""
        print("Preprocessing training data...")
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        
        print("Vectorizing text...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train_processed)
        
        print("Training model...")
        self.model.fit(X_train_vectorized, y_train)
        
        print("Training complete!")
        
    def predict(self, texts):
        """Predict sentiment for new texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        vectorized_texts = self.vectorizer.transform(processed_texts)
        predictions = self.model.predict(vectorized_texts)
        probabilities = self.model.predict_proba(vectorized_texts)
        
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        X_test_vectorized = self.vectorizer.transform(X_test_processed)
        y_pred = self.model.predict(X_test_vectorized)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Negative', 'Positive']))
        
        return accuracy
    
    def save_model(self, path='models/sentiment_model.pkl'):
        """Save trained model and vectorizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer
            }, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/sentiment_model.pkl'):
        """Load trained model and vectorizer"""
        with open(path, 'rb') as f:
            saved_data = pickle.load(f)
            self.model = saved_data['model']
            self.vectorizer = saved_data['vectorizer']
        print(f"Model loaded from {path}")


def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    # Sample reviews (mix of positive and negative)
    reviews = [
        # Positive reviews
        "This product is absolutely amazing! Best purchase I've ever made.",
        "Excellent quality and fast shipping. Highly recommend!",
        "Love it! Exceeded all my expectations.",
        "Outstanding service and great product. Will buy again.",
        "Perfect! Exactly what I was looking for.",
        "Incredible value for money. Very satisfied with this purchase.",
        "This is the best thing I've bought in years. Fantastic!",
        "Awesome product! Works perfectly and looks great.",
        "Very happy with this purchase. Great quality!",
        "Superb! Couldn't ask for anything better.",
        
        # Negative reviews
        "Terrible product. Complete waste of money.",
        "Very disappointed. Poor quality and arrived damaged.",
        "Don't buy this. It broke after one day.",
        "Awful experience. Would not recommend to anyone.",
        "Save your money. This is garbage.",
        "Worst purchase ever. Doesn't work as advertised.",
        "Horrible quality. Requesting a refund immediately.",
        "Completely useless. Very frustrated with this.",
        "Poor customer service and defective product.",
        "Extremely disappointed. This is a scam.",
    ]
    
    # Labels: 1 for positive, 0 for negative
    labels = [1]*10 + [0]*10
    
    return reviews, labels


if __name__ == "__main__":
    print("=" * 60)
    print("Sentiment Analysis Model Training")
    print("=" * 60)
    
    # Create sample dataset
    print("\nCreating sample dataset...")
    reviews, labels = create_sample_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, labels, test_size=0.3, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Initialize and train model
    analyzer = SentimentAnalyzer()
    analyzer.train(X_train, y_train)
    
    # Evaluate model
    analyzer.evaluate(X_test, y_test)
    
    # Save model
    analyzer.save_model('models/sentiment_model.pkl')
    
    # Test with sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions:")
    print("=" * 60)
    
    test_reviews = [
        "This is great! I love it.",
        "Terrible product, very disappointed.",
        "It's okay, nothing special."
    ]
    
    for review in test_reviews:
        predictions, probabilities = analyzer.predict(review)
        sentiment = "Positive" if predictions[0] == 1 else "Negative"
        confidence = probabilities[0][predictions[0]] * 100
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f}%)")
