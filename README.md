---
title: Ebuss Product Recommender
emoji: 🛒
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
---

# Sentiment-Based Product Recommendation System

## Ebuss E-Commerce Platform

A hybrid recommendation system that combines collaborative filtering with sentiment analysis. The idea is simple: don't just recommend what similar users bought - recommend what they actually *liked*.

**Live Demo:** https://huggingface.co/spaces/muralikondapally/SentimentBasedProductRecommendationSystem

---

## What This Project Does

Most recommendation systems just look at purchase patterns. But a product might be selling well while having terrible reviews. This system fixes that by:

1. Finding products similar users liked (User-Based Collaborative Filtering)
2. Analyzing all reviews for those products (Sentiment Analysis with Random Forest)
3. Only recommending products with genuinely positive sentiment

Enter a username → get 5 products that similar users loved AND have good reviews.

---

## The Tech Stack

- **Python 3.11** - because 3.13 broke some dependencies
- **Random Forest** - for sentiment classification (tried XGBoost but it was too big)
- **TF-IDF** - converting review text to features (5000 features, unigrams + bigrams)
- **User-Based CF** - more interpretable than item-based (see below for why)
- **Flask** - simple and gets the job done
- **Docker** - for deployment on Hugging Face Spaces

---

## Challenges I Ran Into

### Class Imbalance (9:1 ratio)

90% of reviews are positive. My first model just predicted "Positive" for everything and got 90% accuracy. Useless.

**Fix:** SMOTE to balance the training data. F1 improved significantly after this.

### XGBoost Was Too Big

XGBoost model was ~200MB. Render's free tier has 512MB RAM total. App kept crashing.

**Fix:** Switched to Random Forest. Best F1-score (~0.95) and smaller footprint.

### NLTK Downloads Failing Silently

Worked locally, failed on Render. No error messages, just broken predictions.

**Fix:** Added try-except around downloads, set `quiet=True`, discovered I needed `punkt_tab` for newer NLTK versions.

### Git LFS Files Not Actually Downloading

Spent hours on this. Models existed but were just LFS pointer files (tiny text files instead of actual model data).

**Fix:** Added `build.sh` with explicit `git lfs pull`. Also added file size checks at startup to catch this early.

### Item-Based CF Had Better Metrics But...

Surprise: Item-Based CF actually had better recall (21% vs 5%) on the test set. But both had very low precision (<2%) due to the 99% matrix sparsity.

**Decision:** Kept User-Based CF anyway because (1) the absolute differences are tiny, (2) sentiment filtering is the real quality gate, and (3) "shoppers like you" makes more sense to users than "similar items".

### TF-IDF Vocabulary Mismatch

Got garbage predictions until I realized I was creating a *new* TF-IDF vectorizer for inference instead of using the trained one. Different vocabulary = wrong features.

**Fix:** Pickle the vectorizer alongside the model. Use the same instance everywhere.

---

## Project Structure

```
├── Sentiment_Based_Product_Recommendation_System.ipynb  # All the analysis
├── model.py                    # Prediction and recommendation logic
├── app.py                      # Flask app
├── templates/
│   └── index.html              # Frontend (Ebuss branded)
├── models/                     # Trained models (run notebook first)
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── user_similarity.pkl
│   └── ...
├── data/
│   └── sample30.csv            # ~30k reviews
├── Dockerfile                  # For HF Spaces
├── requirements.txt
└── README.md
```

---

## How to Run Locally

```bash
# Clone it
git clone https://github.com/muralik31/Capstone--SentimentBasedProductRecommendationSystem.git
cd "Sentiment Based Product Recommendation System"

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the notebook first (generates model files)
jupyter notebook Sentiment_Based_Product_Recommendation_System.ipynb
# Execute all cells

# Then run the app
python app.py
# Visit http://localhost:7860
```

---

## Model Performance

| Model | F1-Score | Notes |
|-------|----------|-------|
| **Random Forest** | 0.95 | Best F1, good size for deployment |
| XGBoost | 0.95 | Similar performance but too big (~200MB) |
| Logistic Regression | 0.93 | Smallest model, good baseline |
| Naive Bayes | 0.90 | Fast but lower recall |

Went with Random Forest - best F1-score and fits in memory.

### Why User-Based CF?

| Approach | Recall | Precision | Choice |
|----------|--------|-----------|--------|
| Item-Based CF | 21% | 1.1% | Better metrics |
| **User-Based CF** | 5% | 0.3% | **Chose this** |

Wait, why pick the "worse" one? Because:
1. Both have <2% precision - neither is actually good at this sparsity level
2. Sentiment filtering is the real quality gate anyway
3. "Users like you bought..." is better UX than "similar items"

---

## API Endpoints

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/` | GET | The UI |
| `/recommend` | POST | Get recommendations (send `{"username": "..."}`) |
| `/api/users` | GET | List of valid usernames for testing |
| `/api/health` | GET | Health check + model file info |

---

## Dataset

~30,000 reviews across 271 products from ~25,000 users. Typical e-commerce stuff - lots of Clorox products, mostly positive reviews, most users only review 1-2 items.

---

## Author

Murali Kondapally  

---
