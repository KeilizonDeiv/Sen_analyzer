"""
Sentiment Analysis Model Trainer with Multi-Emotion Detection
Trains a custom sentiment + emotion classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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
        # Emotion labels: 0=Negative, 1=Neutral, 2=Happy, 3=Angry, 4=Sad, 5=Positive
        self.emotion_map = {
            0: 'Negative',
            1: 'Neutral',
            2: 'Happy',
            3: 'Angry',
            4: 'Sad',
            5: 'Positive'
        }
        
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
        emotion_names = [self.emotion_map[i] for i in range(len(self.emotion_map))]
        print(classification_report(y_test, y_pred, target_names=emotion_names))
        
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
    """Create a sample dataset with emotion labels"""
    # Sample reviews with emotions: 0=Negative, 1=Neutral, 2=Happy, 3=Angry, 4=Sad, 5=Positive
    reviews = [
        # Positive/Happy (2, 5)
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
        
        # Negative/Angry (0, 3)
        "Terrible product. Complete waste of money.",
        "Very disappointed. Poor quality and arrived damaged.",
        "Don't buy this. It broke after one day.",
        "Awful experience. Would not recommend to anyone.",
        "Save your money. This is garbage.",
        "Worst purchase ever. Doesn't work as advertised.",
        "I'm so angry about this purchase. Horrible quality.",
        "This is infuriating! Completely useless.",
        "Extremely frustrated with this product.",
        "Disgusted with this purchase. Absolute disaster.",
        
        # Sad/Disappointed (4)
        "I'm heartbroken by the quality. Really sad about this.",
        "This made me really sad. It was supposed to be special.",
        "Disappointed and sad with how this turned out.",
        "Feeling down about this purchase.",
        
        # Neutral (1)
        "It's okay, nothing special.",
        "Average product, does what it says.",
        "Not great, not terrible.",
        "It's fine, I guess.",
    ]
    
    # Labels: 2=Happy, 5=Positive, 0=Negative, 3=Angry, 4=Sad, 1=Neutral
    labels = [2]*5 + [5]*5 + [0]*5 + [3]*5 + [4]*4 + [1]*4
    
    return reviews, labels


if __name__ == "__main__":
    print("=" * 60)
    print("Sentiment & Emotion Analysis Model Training")
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
        "I'm so angry with this. What a disaster!",
        "It's okay, nothing special.",
        "I'm really sad about this purchase."
    ]
    
    for review in test_reviews:
        predictions, probabilities = analyzer.predict(review)
        emotion = analyzer.emotion_map[predictions[0]]
        confidence = probabilities[0][predictions[0]] * 100
        print(f"\nReview: {review}")
        print(f"Emotion: {emotion} (Confidence: {confidence:.2f}%)")
