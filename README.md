---
title: Ebuss Product Recommender
emoji: 🛒
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

#  Sentiment-Based Product Recommendation System

## Ebuss E-Commerce Platform

An intelligent product recommendation system that combines **collaborative filtering** with **sentiment analysis** to provide personalized and sentiment-aware product recommendations.

---

##  Deployment Link

**[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/ebuss-recommender)**

---

##  Project Overview

This project builds a complete end-to-end ML pipeline for e-commerce product recommendations:

1. **Sentiment Analysis** - Classifies product reviews as positive/negative
2. **Collaborative Filtering** - Recommends products based on user behavior
3. **Sentiment-Filtered Recommendations** - Combines both approaches for better recommendations

---

##  Project Structure

```
Sentiment Based Product Recommendation System/

 Sentiment_Based_Product_Recommendation_System.ipynb  # Main analysis notebook
 model.py                    # ML model and recommendation functions
 app.py                      # Flask application
 requirements.txt            # Python dependencies

 templates/
    index.html             # Web interface

 models/                     # Trained models (generated after running notebook)
    sentiment_model.pkl
    tfidf_vectorizer.pkl
    user_item_matrix.pkl
    user_similarity.pkl
    item_similarity.pkl
    cleaned_data.pkl
    valid_users.pkl

 data/
    sample30.csv           # Dataset (place here)

 Procfile                   # Heroku deployment
 runtime.txt                # Python version for Heroku
 README.md                  # This file
```

---

##  Installation & Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd "Sentiment Based Product Recommendation System"
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 5. Place dataset
Download the `sample30.csv` dataset and place it in the `data/` folder.

---

##  Running the Analysis

### Step 1: Run the Jupyter Notebook
```bash
jupyter notebook Sentiment_Based_Product_Recommendation_System.ipynb
```

Execute all cells to:
- Perform EDA and data cleaning
- Preprocess text data
- Train sentiment analysis models (Logistic Regression, Random Forest, XGBoost, Naive Bayes)
- Build recommendation systems (User-based and Item-based CF)
- Export trained models to `models/` directory

### Step 2: Run the Flask Application
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

##  Heroku Deployment

### 1. Login to Heroku
```bash
heroku login
```

### 2. Create Heroku app
```bash
heroku create your-app-name
```

### 3. Deploy
```bash
git add .
git commit -m "Initial deployment"
git push heroku main
```

### 4. Open app
```bash
heroku open
```

---

##  Model Performance

### Sentiment Analysis Models Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.90 | ~0.92 | ~0.95 | ~0.93 | ~0.95 |
| Random Forest | ~0.88 | ~0.91 | ~0.93 | ~0.92 | ~0.94 |
| XGBoost | ~0.89 | ~0.91 | ~0.94 | ~0.92 | ~0.95 |
| Naive Bayes | ~0.85 | ~0.88 | ~0.92 | ~0.90 | ~0.92 |

*Note: Actual values depend on your dataset run*

### Recommendation System Comparison

| System | Precision@20 | Recall@20 | F1-Score |
|--------|--------------|-----------|----------|
| User-Based CF | Varies | Varies | Varies |
| Item-Based CF | Varies | Varies | Varies |

---

##  Features

-  Comprehensive EDA with visualizations
-  Text preprocessing (tokenization, lemmatization, stopword removal)
-  TF-IDF feature extraction with n-grams
-  Class imbalance handling with SMOTE
-  Hyperparameter tuning with GridSearchCV
-  Four sentiment analysis models
-  Two recommendation systems
-  Sentiment-filtered recommendations
-  Beautiful web interface
-  Heroku deployment ready

---

##  Technologies Used

- **Python 3.10**
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn** - ML models
- **XGBoost** - Gradient boosting
- **NLTK** - NLP preprocessing
- **Flask** - Web framework
- **Gunicorn** - WSGI server
- **Heroku** - Cloud deployment

---

##  API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with search interface |
| `/recommend` | POST | Get recommendations for a user |
| `/api/users` | GET | Get sample usernames |
| `/api/health` | GET | Health check |

---

## ‍ Author

Machine Learning Engineer, Ebuss

---

##  License

This project is for educational purposes.

---

##  Acknowledgments

- Dataset inspired by Kaggle e-commerce reviews
- Built as part of ML Engineering project

