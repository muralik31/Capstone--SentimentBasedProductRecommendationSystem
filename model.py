# Sentiment + Recommendation logic for Ebuss

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

# Download NLTK data quietly
for resource in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except:
        nltk.download(resource, quiet=True)


class SentimentRecommender:
    """Hybrid recommender: CF + sentiment filtering"""
    
    def __init__(self):
        self.sentiment_model = None
        self.tfidf = None
        self.user_item_matrix = None
        self.user_sim = None
        self.item_sim = None
        self.data = None
        self.users = None
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_models(self, path='models/'):
        """Load pickled models from disk"""
        if not os.path.exists(path):
            print(f"ERROR: {path} doesn't exist!")
            return False
        
        # Check for LFS pointer files (caused issues on Render)
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
            self.sentiment_model = joblib.load(path + 'sentiment_model.pkl')
            self.tfidf = joblib.load(path + 'tfidf_vectorizer.pkl')
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
        """Clean text - must match training preprocessing exactly"""
        if pd.isna(text): return ""
        
        text = str(text).lower()
        try: text = contractions.fix(text)
        except: pass
        
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        try: tokens = word_tokenize(text)
        except: tokens = text.split()
        
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens 
                  if t not in self.stopwords and len(t) > 2]
        return ' '.join(tokens)
    
    def predict_sentiment(self, texts):
        """Get sentiment probability (positive class) for texts"""
        if not self.sentiment_model:
            raise ValueError("Models not loaded!")
        processed = [self.preprocess(t) for t in texts]
        features = self.tfidf.transform(processed)
        return self.sentiment_model.predict_proba(features)[:, 1]
    
    def get_user_recommendations(self, user, n=20):
        """User-based CF: find similar users, get their liked products"""
        if self.user_item_matrix is None or user not in self.user_item_matrix.index:
            return []
        
        sim_users = self.user_sim[user].drop(user).sort_values(ascending=False)
        user_rated = self.user_item_matrix.loc[user]
        already_rated = user_rated[user_rated > 0].index.tolist()
        
        recs = {}
        for prod in self.user_item_matrix.columns:
            if prod in already_rated: continue
            
            prod_ratings = self.user_item_matrix[prod]
            rated = prod_ratings[prod_ratings > 0]
            if len(rated) == 0: continue
            
            sims = sim_users.reindex(rated.index).fillna(0)
            if sims.sum() > 0:
                recs[prod] = (rated * sims).sum() / sims.sum()
        
        sorted_recs = sorted(recs.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_recs[:n]]

    def get_item_recommendations(self, user, n=20, k=10):
        """Item-based CF: find similar items to what user liked"""
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
        Main function: get CF recommendations, filter by sentiment.
        Returns top 5 products with best sentiment scores.
        """
        if self.data is None or user not in self.users:
            return None
        
        # Get CF recommendations (user-based by default)
        if use_item:
            cf_recs = self.get_item_recommendations(user, n_cf)
        else:
            cf_recs = self.get_user_recommendations(user, n_cf)
        
        if not cf_recs: return []
        
        # Score each product by sentiment
        results = []
        for prod in cf_recs:
            reviews = self.data[self.data['name'] == prod]['reviews_text'].tolist()
            if not reviews: continue
            
            probs = self.predict_sentiment(reviews)
            results.append({
                'product': prod,
                'sentiment_score': np.mean(probs),
                'positive_ratio': np.mean(probs > 0.5),
                'avg_rating': self.data[self.data['name'] == prod]['reviews_rating'].mean(),
                'num_reviews': len(reviews)
            })
        
        results.sort(key=lambda x: x['sentiment_score'], reverse=True)
        return results[:n_final]
    
    def is_valid_user(self, user):
        return self.users and user in self.users
    
    def get_sample_users(self, n=10):
        return self.users[:n] if self.users else []


# Global instance
recommender = SentimentRecommender()

def initialize_models(path='models/'):
    return recommender.load_models(path)

def get_recommendations(user):
    return recommender.get_sentiment_recommendations(user)

def check_user(user):
    return recommender.is_valid_user(user)

def get_sample_users(n=10):
    return recommender.get_sample_users(n)
