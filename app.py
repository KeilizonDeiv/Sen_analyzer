"""
Flask Web Application for Sentiment & Emotion Analysis
Provides a web interface with advanced features: emotion detection, caching, logging, and export
"""

from flask import Flask, render_template, request, jsonify, send_file
import pickle
import os
from train_model import SentimentAnalyzer
from datetime import datetime
import json
import logging
from functools import lru_cache
import io
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the trained model
analyzer = SentimentAnalyzer()

# Store analysis history
history = []

# Prediction cache
prediction_cache = {}

def load_model():
    """Load the trained model at startup"""
    try:
        model_path = 'models/sentiment_model.pkl'
        if os.path.exists(model_path):
            analyzer.load_model(model_path)
            logger.info("Model loaded successfully!")
        else:
            logger.error("No trained model found. Please run train_model.py first.")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    """Render the main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return jsonify({'error': 'Failed to load interface'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze emotion of submitted text with caching"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            logger.warning("Empty text submission")
            return jsonify({'error': 'Please enter some text to analyze'}), 400
        
        # Check cache first
        text_hash = hash(text)
        if text_hash in prediction_cache:
            logger.info(f"Cache hit for text: {text[:50]}...")
            return jsonify(prediction_cache[text_hash])
        
        # Perform prediction
        predictions, probabilities = analyzer.predict(text)
        emotion_label = predictions[0]
        emotion = analyzer.emotion_map.get(emotion_label, 'Unknown')
        confidence = float(probabilities[0][emotion_label] * 100)
        
        # Get all emotion probabilities for visualization
        emotion_details = {}
        for idx, emotion_name in analyzer.emotion_map.items():
            emotion_details[emotion_name] = float(probabilities[0][idx] * 100)
        
        # Store in history
        result = {
            'text': text,
            'emotion': emotion,
            'confidence': round(confidence, 2),
            'emotion_details': emotion_details,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        history.append(result)
        prediction_cache[text_hash] = result
        
        # Keep only last 50 results
        if len(history) > 50:
            old_result = history.pop(0)
            # Remove from cache if it's the oldest
            if hash(old_result['text']) in prediction_cache:
                del prediction_cache[hash(old_result['text'])]
        
        logger.info(f"Analysis complete: {emotion} (confidence: {confidence:.2f}%)")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/history')
def get_history():
    """Get analysis history"""
    try:
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return jsonify({'error': 'Failed to retrieve history'}), 500

@app.route('/stats')
def get_stats():
    """Get statistics about analyzed reviews with emotion breakdown"""
    try:
        if not history:
            return jsonify({
                'total': 0,
                'emotion_counts': {emotion: 0 for emotion in analyzer.emotion_map.values()},
                'avg_confidence': 0
            })
        
        total = len(history)
        emotion_counts = {emotion: 0 for emotion in analyzer.emotion_map.values()}
        
        for h in history:
            emotion_counts[h['emotion']] += 1
        
        avg_confidence = sum(h['confidence'] for h in history) / total
        
        stats = {
            'total': total,
            'emotion_counts': emotion_counts,
            'avg_confidence': round(avg_confidence, 2)
        }
        logger.info(f"Stats retrieved: {stats}")
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error calculating stats: {str(e)}")
        return jsonify({'error': 'Failed to calculate statistics'}), 500

@app.route('/export', methods=['GET'])
def export_results():
    """Export analysis history as CSV"""
    try:
        if not history:
            return jsonify({'error': 'No data to export'}), 400
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Timestamp', 'Text', 'Emotion', 'Confidence (%)', 'Details'])
        
        # Write data
        for result in history:
            emotion_str = ', '.join([f"{k}: {v:.1f}%" for k, v in result['emotion_details'].items()])
            writer.writerow([
                result['timestamp'],
                result['text'],
                result['emotion'],
                result['confidence'],
                emotion_str
            ])
        
        # Convert to bytes
        output.seek(0)
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        logger.info(f"Exported {len(history)} results to CSV")
        return send_file(mem, mimetype='text/csv', as_attachment=True, download_name=f'sentiment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear prediction cache and history"""
    try:
        global history, prediction_cache
        history = []
        prediction_cache = {}
        logger.info("Cache and history cleared")
        return jsonify({'message': 'Cache cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({'error': 'Failed to clear cache'}), 500

if __name__ == '__main__':
    load_model()
    logger.info("Flask app starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)
