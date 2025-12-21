---
title: Ebuss Product Recommender
emoji: 🛒
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# Sentiment-Based Product Recommendation System

## Ebuss E-Commerce Platform

An intelligent product recommendation system that combines **collaborative filtering** with **sentiment analysis** to provide personalized and sentiment-aware product recommendations.

---

## Deployment Link

**[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/muralikondapally/SentimentBasedProductRecommendationSystem)**

---

## Project Overview

This project builds a complete end-to-end ML pipeline for e-commerce product recommendations:

1. **Sentiment Analysis** - Classifies product reviews as positive/negative using ML
2. **Collaborative Filtering** - Recommends products based on user behavior patterns
3. **Hybrid Recommendations** - Filters CF results through sentiment scores for better quality

---

## Challenges Encountered

Building this project wasn't straightforward - here are some real issues I ran into:

### 1. Class Imbalance Was Severe (~9:1 ratio)

The dataset had way more positive reviews than negative ones. My initial Logistic Regression model just predicted "Positive" for everything and still got 90% accuracy - completely useless for actually identifying negative sentiment.

**Solution:** Used SMOTE (Synthetic Minority Over-sampling) to balance the training data. F1-score jumped from 0.88 to 0.93 after this fix.

### 2. XGBoost Model Too Large for Deployment

I initially picked XGBoost as the best model (slightly better F1-score), but the serialized model was ~200MB. Render's free tier only has 512MB RAM, so the app kept crashing on startup.

**Solution:** Switched to Random Forest which gave similar performance (~0.92 F1) but with a smaller memory footprint. Still had to use Git LFS for model files.

### 3. NLTK Downloads Failing on Cloud Platforms

The NLTK data downloads (`punkt`, `stopwords`, `wordnet`) worked fine locally but silently failed on Render and Hugging Face Spaces during container startup.

**Solution:** Added try-except blocks around each download and set `quiet=True`. Also discovered I needed `punkt_tab` for newer NLTK versions - wasn't documented anywhere.

### 4. Git LFS Files Not Downloading on Render

Spent hours debugging why models wouldn't load. Turns out Render clones repos but doesn't automatically run `git lfs pull`. The model files existed but were just tiny LFS pointer files.

**Solution:** Added a `build.sh` script with explicit `git lfs pull` command, and added diagnostic code to check file sizes at startup.

### 5. Item-Based vs User-Based CF Decision

Initially implemented both, but item-based CF performed poorly on our sparse matrix (most users rate only 5-10 products). The item-item similarity matrix had too many near-zero values.

**Solution:** Kept both implementations but defaulted to user-based CF which gave more diverse recommendations.

### 6. TF-IDF Feature Mismatch During Inference

Got weird predictions in production until I realized I was creating a new TF-IDF vectorizer instead of using the one from training. The vocabulary didn't match, so features were completely wrong.

**Solution:** Pickle the TF-IDF vectorizer along with the model and use the same instance for inference.

---

## Project Structure

```
Sentiment Based Product Recommendation System/
├── Sentiment_Based_Product_Recommendation_System.ipynb  # Main analysis notebook
├── model.py                    # ML model and recommendation functions
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Web interface (Ebuss branded)
├── models/                     # Trained models (generated after running notebook)
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── user_item_matrix.pkl
│   ├── user_similarity.pkl
│   ├── item_similarity.pkl
│   ├── cleaned_data.pkl
│   └── valid_users.pkl
├── data/
│   └── sample30.csv            # Dataset
├── Dockerfile                  # For Hugging Face Spaces
├── Procfile                    # Heroku deployment
├── runtime.txt                 # Python version
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/muralik31/Capstone--SentimentBasedProductRecommendationSystem.git
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

### 4. Place dataset
Download the `sample30.csv` dataset and place it in the `data/` folder.

---

## Running the Analysis

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

Visit `http://localhost:7860` in your browser.

---

## Model Performance

### Sentiment Analysis Models Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.90 | ~0.92 | ~0.95 | ~0.93 | ~0.95 |
| **Random Forest** | ~0.88 | ~0.91 | ~0.93 | ~0.92 | ~0.94 |
| XGBoost | ~0.89 | ~0.91 | ~0.94 | ~0.92 | ~0.95 |
| Naive Bayes | ~0.85 | ~0.88 | ~0.92 | ~0.90 | ~0.92 |

*Random Forest selected for deployment (good F1-score + reasonable model size)*

### Recommendation System Selection

| System | Chosen | Reason |
|--------|--------|--------|
| **User-Based CF** | Yes | Better performance on sparse matrix, more diverse recommendations |
| Item-Based CF | No | Item similarity matrix too sparse, recommendations less varied |

---

## How It Works

1. **User enters username** in the web interface
2. **User-Based CF** finds 20 products from similar users that this user hasn't rated
3. **Sentiment Analysis** scores all reviews for each of those 20 products
4. **Top 5 products** with highest positive sentiment ratio are returned

This hybrid approach ensures recommendations are not just relevant (CF) but also genuinely well-reviewed (sentiment).

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with search interface |
| `/recommend` | POST | Get recommendations for a user |
| `/api/users` | GET | Get sample usernames |
| `/api/health` | GET | Health check + model file info |

---

## Technologies Used

- **Python 3.11**
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn** - ML models (Random Forest, Logistic Regression)
- **XGBoost** - Gradient boosting (tested, not deployed)
- **NLTK** - NLP preprocessing
- **Flask** - Web framework
- **Gunicorn** - WSGI server
- **Docker** - Containerization
- **Hugging Face Spaces** - Cloud deployment

---

## Author

Murali Kondapally  
ML Engineering Capstone Project

---

## License

This project is for educational purposes.

---

## Acknowledgments

- Dataset inspired by Kaggle e-commerce reviews
- Built as part of ML Engineering coursework
