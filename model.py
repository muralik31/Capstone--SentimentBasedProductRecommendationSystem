# model.py - Sentiment + Recommendation Engine for Ebuss
# -------------------------------------------------------
# This module handles the core ML logic: sentiment classification + collaborative filtering.
# I went with a class-based approach to keep all the loaded models in one place -
# made debugging much easier when things broke during deployment.

import pandas as pd
import numpy as np
import re
import joblib
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions

# Download NLTK resources quietly - learned the hard way that this fails silently on some hosts
# so wrapping each in try-except. The punkt_tab is new in recent NLTK versions.
for resource in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except:
        nltk.download(resource, quiet=True)


class SentimentRecommender:
    """
    Main recommender class - combines collaborative filtering with sentiment analysis.
    
    Why this hybrid approach?
    Pure CF gives "popular" items but doesn't account for review quality.
    By filtering CF results through sentiment scores, we surface products
    that users ACTUALLY liked, not just products that got lots of reviews.
    """
    
    def __init__(self):
        self.sentiment_model = None
        self.tfidf = None
        self.user_item_matrix = None
        self.user_sim = None  # user-user similarity matrix
        self.item_sim = None  # item-item similarity matrix
        self.data = None
        self.users = None
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_models(self, path='models/'):
        """
        Load all pickled models from disk.
        
        Note: Had to add LFS pointer detection because models weren't loading on Render.
        Turns out Git LFS files weren't being pulled during build - the files existed
        but were just tiny pointer files, not the actual model binaries.
        """
        if not os.path.exists(path):
            print(f"ERROR: {path} doesn't exist!")
            return False
        
        # LFS pointer check - these are text files < 200 bytes that start with "version https://git-lfs"
        # Wasted 3 hours debugging "unpickling error" before realizing this was the issue
        print(f"Loading models from {path}...")
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            sz = os.path.getsize(fpath)
            if sz < 200:
                try:
                    with open(fpath, 'r') as fp:
                        if fp.read(30).startswith('version https://git-lfs'):
                            print(f"WARNING: {f} is LFS pointer! Run git lfs pull")
                            return False
                except: pass
            print(f"  {f}: {sz/(1024*1024):.1f}MB")
        
        try:
            # Load the sentiment model - using joblib because sklearn recommends it for their models
            # (pickle works too but joblib handles numpy arrays more efficiently)
            self.sentiment_model = joblib.load(path + 'sentiment_model.pkl')
            
            # TF-IDF vectorizer from training - MUST use the same one for inference
            # otherwise the feature dimensions won't match and predictions will be garbage
            self.tfidf = joblib.load(path + 'tfidf_vectorizer.pkl')
            
            # Pre-computed matrices from notebook - these take forever to build so we pickle them
            self.user_item_matrix = pd.read_pickle(path + 'user_item_matrix.pkl')
            self.user_sim = pd.read_pickle(path + 'user_similarity.pkl')
            self.item_sim = pd.read_pickle(path + 'item_similarity.pkl')
            self.data = pd.read_pickle(path + 'cleaned_data.pkl')
            
            with open(path + 'valid_users.pkl', 'rb') as f:
                self.users = pickle.load(f)
            
            print(f"Loaded! {len(self.users)} users, {self.user_item_matrix.shape[1]} products")
            return True
        except Exception as e:
            print(f"Error loading: {e}")
            import traceback; traceback.print_exc()
            return False
    
    def preprocess(self, text):
        """
        Clean and normalize text for sentiment analysis.
        
        This mirrors EXACTLY what was done during training (see notebook cell 23).
        If preprocessing differs even slightly, the TF-IDF features won't align
        and the model predictions become unreliable.
        """
        if pd.isna(text): return ""
        
        text = str(text).lower()
        
        # Expand contractions first - "don't" -> "do not"
        # This helps because our TF-IDF was trained on expanded text
        try: text = contractions.fix(text)
        except: pass
        
        # Strip URLs, HTML tags, and non-alphabetic chars
        # Keeping only letters because numbers/symbols don't help sentiment much
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())  # normalize whitespace
        
        # Tokenize and lemmatize - reduces "running", "runs", "ran" -> "run"
        # Helps the model generalize better with limited vocab
        try: tokens = word_tokenize(text)
        except: tokens = text.split()  # fallback if NLTK fails
        
        # Filter out stopwords and very short tokens (usually noise)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens 
                  if t not in self.stopwords and len(t) > 2]
        return ' '.join(tokens)
    
    def predict_sentiment(self, texts):
        """
        Get sentiment probabilities for a list of review texts.
        Returns probability of POSITIVE sentiment (class 1).
        
        Using predict_proba instead of predict because we want the confidence score,
        not just 0/1 - this lets us rank products by sentiment quality.
        """
        if not self.sentiment_model:
            raise ValueError("Models not loaded!")
        processed = [self.preprocess(t) for t in texts]
        features = self.tfidf.transform(processed)
        return self.sentiment_model.predict_proba(features)[:, 1]
    
    def get_user_recommendations(self, user, n=20):
        """
        User-based collaborative filtering.
        
        The idea: find users similar to our target user (based on rating patterns),
        then recommend items those similar users liked but our user hasn't seen.
        
        Chose this over item-based for Ebuss because our user-item matrix is sparse -
        users typically rate only 5-10 products. User-based handles this better
        when there are more users than items, which is our case (20k users, ~200 products).
        """
        if self.user_item_matrix is None or user not in self.user_item_matrix.index:
            return []
        
        # Get similarity scores with all other users, excluding self
        sim_users = self.user_sim[user].drop(user).sort_values(ascending=False)
        
        # Find what our user has already rated (don't recommend these)
        user_rated = self.user_item_matrix.loc[user]
        already_rated = user_rated[user_rated > 0].index.tolist()
        
        # For each unrated product, compute weighted average rating
        # Weight = similarity to users who rated that product
        recs = {}
        for prod in self.user_item_matrix.columns:
            if prod in already_rated: continue
            
            prod_ratings = self.user_item_matrix[prod]
            rated = prod_ratings[prod_ratings > 0]
            if len(rated) == 0: continue
            
            # Weighted average: sum(rating * similarity) / sum(similarity)
            sims = sim_users.reindex(rated.index).fillna(0)
            if sims.sum() > 0:
                recs[prod] = (rated * sims).sum() / sims.sum()
        
        sorted_recs = sorted(recs.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_recs[:n]]

    def get_item_recommendations(self, user, n=20, k=10):
        """
        Item-based collaborative filtering (alternative approach).
        
        Different philosophy: instead of finding similar users, find similar items.
        For each unrated item, look at items the user HAS rated and check
        how similar those are to the candidate item.
        
        Kept this as an option even though user-based won in my experiments -
        might be useful if the user base grows significantly.
        """
        if self.user_item_matrix is None or user not in self.user_item_matrix.index:
            return []
        
        user_rated = self.user_item_matrix.loc[user]
        rated_prods = user_rated[user_rated > 0]
        unrated = self.user_item_matrix.columns[user_rated == 0]
        
        if len(rated_prods) == 0: return []
        
        recs = {}
        for prod in unrated:
            sims, ratings = [], []
            for rated_p, rating in rated_prods.items():
                if prod in self.item_sim.columns and rated_p in self.item_sim.index:
                    s = self.item_sim.loc[rated_p, prod]
                    if s > 0:
                        sims.append(s)
                        ratings.append(rating)
            
            if not sims: continue
            
            # Use only top-k most similar items to avoid noise from weakly similar items
            if len(sims) > k:
                idx = np.argsort(sims)[-k:]
                sims = [sims[i] for i in idx]
                ratings = [ratings[i] for i in idx]
            
            if sum(sims) > 0:
                recs[prod] = np.average(ratings, weights=sims)
        
        sorted_recs = sorted(recs.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_recs[:n]]
    
    def get_sentiment_recommendations(self, user, n_cf=20, n_final=5, use_item=False):
        """
        THE MAIN FUNCTION - Get recommendations filtered by sentiment.
        
        This is where the magic happens:
        1. Get top 20 products from collaborative filtering
        2. For each product, analyze ALL its reviews using our sentiment model
        3. Rank by average sentiment score
        4. Return top 5
        
        Why 20 -> 5? Gives the sentiment model enough candidates to find
        truly positive products, while still being fast enough for real-time use.
        
        Using user-based CF by default (use_item=False) based on notebook analysis
        showing it performed better on our sparse matrix.
        """
        if self.data is None or user not in self.users:
            return None
        
        # Step 1: Get CF recommendations (defaulting to user-based per notebook conclusion)
        if use_item:
            cf_recs = self.get_item_recommendations(user, n_cf)
        else:
            cf_recs = self.get_user_recommendations(user, n_cf)
        
        if not cf_recs: return []
        
        # Step 2: Score each recommendation by sentiment
        results = []
        for prod in cf_recs:
            reviews = self.data[self.data['name'] == prod]['reviews_text'].tolist()
            if not reviews: continue
            
            # Get sentiment probability for each review
            probs = self.predict_sentiment(reviews)
            avg_sent = np.mean(probs)  # average sentiment score
            pos_ratio = np.mean(probs > 0.5)  # % of reviews classified as positive
            avg_rating = self.data[self.data['name'] == prod]['reviews_rating'].mean()
            
            results.append({
                'product': prod,
                'sentiment_score': avg_sent,
                'positive_ratio': pos_ratio,
                'avg_rating': avg_rating,
                'num_reviews': len(reviews)
            })
        
        # Step 3: Sort by sentiment and return top n
        results.sort(key=lambda x: x['sentiment_score'], reverse=True)
        return results[:n_final]
    
    def is_valid_user(self, user):
        """Check if user exists in our dataset"""
        return self.users and user in self.users
    
    def get_sample_users(self, n=10):
        """Get sample usernames for the UI dropdown"""
        return self.users[:n] if self.users else []


# Global instance - created once when module is imported
# Flask will use this single instance across requests (no need to reload models)
recommender = SentimentRecommender()

def initialize_models(path='models/'):
    """Called once at app startup to load all models into memory"""
    return recommender.load_models(path)

def get_recommendations(user):
    """Convenience wrapper for Flask routes"""
    return recommender.get_sentiment_recommendations(user)

def check_user(user):
    """Validate username before processing"""
    return recommender.is_valid_user(user)

def get_sample_users(n=10):
    """Get sample users for the UI"""
    return recommender.get_sample_users(n)
