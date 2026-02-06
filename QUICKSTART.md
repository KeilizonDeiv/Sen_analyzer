# 🚀 Quick Start Guide - Sentiment Analyzer

## Get Started in 3 Steps:

### Step 1: Install Dependencies
```bash
cd sentiment_analyzer
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_model.py
```
Expected output:
```
Model Accuracy: 0.85+
Model saved to models/sentiment_model.pkl
```

### Step 3: Run the Application
```bash
python app.py
```
Open browser to: **http://localhost:5000**

---

## What You'll See:

✅ Beautiful web interface for sentiment analysis  
✅ Real-time predictions with confidence scores  
✅ Statistics dashboard showing analysis trends  
✅ History of recent predictions  

## Test It Out:

Try these sample reviews:

**Positive:**
- "This is amazing! Exactly what I needed. Highly recommend!"

**Negative:**
- "Terrible product. Complete waste of money."

**Neutral:**
- "It's okay, does the job but nothing special."

---

## Troubleshooting:

**Issue:** ModuleNotFoundError  
**Solution:** Run `pip install -r requirements.txt`

**Issue:** NLTK data not found  
**Solution:** The script will auto-download required data on first run

**Issue:** Port 5000 already in use  
**Solution:** Change port in app.py: `app.run(port=5001)`

---

## Next Steps for Your Resume:

1. **Upload to GitHub:**
   - Create a new repository
   - Add this project with good commit messages
   - Include the README.md for documentation

2. **Enhance Your Resume:**
   - Add to "Projects" section
   - Mention: ML, NLP, Flask, scikit-learn, NLTK
   - Include GitHub link

3. **Improve the Project:**
   - Train on larger dataset (10,000+ reviews)
   - Add more features (export reports, batch analysis)
   - Deploy to Heroku or AWS for live demo

4. **Interview Talking Points:**
   - Explain the preprocessing pipeline
   - Discuss TF-IDF vectorization
   - Describe model evaluation metrics
   - Talk about deployment considerations

---

**Pro Tip:** When discussing this in interviews, focus on:
- Problem: Need to analyze customer sentiment at scale
- Solution: Built custom ML model with web interface
- Impact: Enables real-time sentiment analysis with 85%+ accuracy
