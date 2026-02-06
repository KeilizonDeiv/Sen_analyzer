"""
Flask Web Application for Sentiment Analysis
Provides a web interface for analyzing customer review sentiment
"""

from flask import Flask, render_template, request, jsonify
import pickle
import os
from train_model import SentimentAnalyzer
from datetime import datetime
import json

app = Flask(__name__)

# Load the trained model
analyzer = SentimentAnalyzer()

# Store analysis history
history = []

def load_model():
    """Load the trained model at startup"""
    model_path = 'models/sentiment_model.pkl'
    if os.path.exists(model_path):
        analyzer.load_model(model_path)
        print("Model loaded successfully!")
    else:
        print("No trained model found. Please run train_model.py first.")

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment of submitted text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'Please enter some text to analyze'}), 400
        
        # Perform prediction
        predictions, probabilities = analyzer.predict(text)
        
        sentiment = "Positive" if predictions[0] == 1 else "Negative"
        confidence = float(probabilities[0][predictions[0]] * 100)
        
        # Store in history
        result = {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        history.append(result)
        
        # Keep only last 50 results
        if len(history) > 50:
            history.pop(0)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    """Get analysis history"""
    return jsonify(history)

@app.route('/stats')
def get_stats():
    """Get statistics about analyzed reviews"""
    if not history:
        return jsonify({
            'total': 0,
            'positive': 0,
            'negative': 0,
            'avg_confidence': 0
        })
    
    total = len(history)
    positive = sum(1 for h in history if h['sentiment'] == 'Positive')
    negative = total - positive
    avg_confidence = sum(h['confidence'] for h in history) / total
    
    return jsonify({
        'total': total,
        'positive': positive,
        'negative': negative,
        'avg_confidence': round(avg_confidence, 2)
    })

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
