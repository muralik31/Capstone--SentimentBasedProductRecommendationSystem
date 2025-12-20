"""
Sentiment-Based Product Recommendation System
Model Module

This module contains:
1. Text preprocessing functions
2. Sentiment analysis model integration
3. User-based collaborative filtering recommendation system
4. Sentiment-filtered recommendation function

Author: Machine Learning Engineer, Ebuss
Date: December 2024
"""

import pandas as pd
import numpy as np
import re
import joblib
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
from sklearn.metrics.pairwise import cosine_similarity


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass


class SentimentRecommender:
    """
    A class that combines sentiment analysis with collaborative filtering
    to provide sentiment-aware product recommendations.
    """
    
    def __init__(self):
        """Initialize the recommender system by loading pre-trained models."""
        self.sentiment_model = None
        self.tfidf_vectorizer = None
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.cleaned_data = None
        self.valid_users = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_models(self, model_path='models/'):
        """
        Load all pre-trained models and data from the specified path.
        
        Args:
            model_path (str): Path to the models directory
        """
        try:
            # Load sentiment model
            self.sentiment_model = joblib.load(f'{model_path}sentiment_model.pkl')
            print(" Sentiment model loaded")
            
            # Load TF-IDF vectorizer
            self.tfidf_vectorizer = joblib.load(f'{model_path}tfidf_vectorizer.pkl')
            print(" TF-IDF vectorizer loaded")
            
            # Load user-item matrix
            self.user_item_matrix = pd.read_pickle(f'{model_path}user_item_matrix.pkl')
            print(" User-Item matrix loaded")
            
            # Load user similarity matrix
            self.user_similarity = pd.read_pickle(f'{model_path}user_similarity.pkl')
            print(" User similarity matrix loaded")
            
            # Load item similarity matrix (for backup)
            self.item_similarity = pd.read_pickle(f'{model_path}item_similarity.pkl')
            print(" Item similarity matrix loaded")
            
            # Load cleaned data
            self.cleaned_data = pd.read_pickle(f'{model_path}cleaned_data.pkl')
            print(" Cleaned data loaded")
            
            # Load valid users list
            with open(f'{model_path}valid_users.pkl', 'rb') as f:
                self.valid_users = pickle.load(f)
            print(" Valid users list loaded")
            
            print(f"\n System Stats:")
            print(f"   - Total Users: {len(self.valid_users)}")
            print(f"   - Total Products: {self.user_item_matrix.shape[1]}")
            
            return True
            
        except Exception as e:
            print(f" Error loading models: {str(e)}")
            return False
    
    def preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Expand contractions
        try:
            text = contractions.fix(text)
        except:
            pass
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenization
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                  for token in tokens 
                  if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def predict_sentiment(self, texts):
        """
        Predict sentiment for given texts.
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            np.array: Array of sentiment probabilities (positive)
        """
        if self.sentiment_model is None or self.tfidf_vectorizer is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform using TF-IDF
        features = self.tfidf_vectorizer.transform(processed_texts)
        
        # Predict probabilities
        probabilities = self.sentiment_model.predict_proba(features)[:, 1]
        
        return probabilities
    
    def get_user_based_recommendations(self, username, n_recommendations=20):
        """
        Get product recommendations using user-based collaborative filtering.

        Args:
            username (str): Username to get recommendations for
            n_recommendations (int): Number of recommendations to return

        Returns:
            list: List of recommended product names
        """
        if self.user_item_matrix is None or self.user_similarity is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        if username not in self.user_item_matrix.index:
            return []

        # Get similar users
        similar_users = self.user_similarity[username].drop(username).sort_values(ascending=False)

        # Get products the user has already rated
        user_rated = self.user_item_matrix.loc[username]
        rated_products = user_rated[user_rated > 0].index.tolist()

        # Calculate weighted average of similar users' ratings
        recommendations = {}

        for product in self.user_item_matrix.columns:
            if product in rated_products:
                continue

            # Get ratings from similar users who rated this product
            product_ratings = self.user_item_matrix[product]
            rated_by_similar = product_ratings[product_ratings > 0]

            if len(rated_by_similar) == 0:
                continue

            # Calculate weighted average
            similarities = similar_users.reindex(rated_by_similar.index).fillna(0)
            if similarities.sum() > 0:
                weighted_avg = (rated_by_similar * similarities).sum() / similarities.sum()
                recommendations[product] = weighted_avg

        # Sort and return top N
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [product for product, score in sorted_recs[:n_recommendations]]

    def get_item_based_recommendations(self, username, n_recommendations=20, k=10):
        """
        Get product recommendations using item-based collaborative filtering.

        This approach:
        1. Finds products the user has already rated
        2. For each unrated product, finds K most similar rated products
        3. Predicts rating as weighted average of similar products' ratings

        Args:
            username (str): Username to get recommendations for
            n_recommendations (int): Number of recommendations to return
            k (int): Number of similar items to consider for prediction

        Returns:
            list: List of recommended product names sorted by predicted rating
        """
        if self.user_item_matrix is None or self.item_similarity is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        if username not in self.user_item_matrix.index:
            return []

        # Get products the user has already rated
        user_rated = self.user_item_matrix.loc[username]
        rated_products = user_rated[user_rated > 0]
        unrated_products = self.user_item_matrix.columns[user_rated == 0]

        if len(rated_products) == 0:
            return []

        # Calculate predicted ratings for unrated products
        recommendations = {}

        for product in unrated_products:
            # Get similarities between this product and all rated products
            similarities = []
            ratings = []

            for rated_product, rating in rated_products.items():
                # Get similarity score
                if product in self.item_similarity.columns and rated_product in self.item_similarity.index:
                    sim = self.item_similarity.loc[rated_product, product]
                    if sim > 0:  # Only consider positive similarities
                        similarities.append(sim)
                        ratings.append(rating)

            if len(similarities) == 0:
                continue

            # Take top K most similar products
            if len(similarities) > k:
                top_k_idx = np.argsort(similarities)[-k:]
                similarities = [similarities[i] for i in top_k_idx]
                ratings = [ratings[i] for i in top_k_idx]

            # Calculate weighted average rating
            if sum(similarities) > 0:
                predicted_rating = np.average(ratings, weights=similarities)
                recommendations[product] = predicted_rating

        # Sort by predicted rating and return top N
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [product for product, score in sorted_recs[:n_recommendations]]
    
    def get_sentiment_filtered_recommendations(self, username, n_initial=20, n_final=5, use_item_based=True):
        """
        Get top N recommendations filtered by sentiment analysis.

        This method:
        1. Gets 20 recommendations from collaborative filtering
        2. Analyzes sentiment of reviews for each recommended product
        3. Returns top 5 products with best sentiment scores

        Args:
            username (str): Username to get recommendations for
            n_initial (int): Number of initial recommendations from CF
            n_final (int): Number of final recommendations after sentiment filtering
            use_item_based (bool): If True, use item-based CF; if False, use user-based CF

        Returns:
            list: List of dictionaries containing product info and sentiment scores
        """
        if self.cleaned_data is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        # Check if user exists
        if username not in self.valid_users:
            return None

        # Get initial recommendations from collaborative filtering
        if use_item_based:
            initial_recommendations = self.get_item_based_recommendations(username, n_initial)
        else:
            initial_recommendations = self.get_user_based_recommendations(username, n_initial)
        
        if len(initial_recommendations) == 0:
            return []
        
        # Calculate sentiment scores for each recommended product
        product_sentiments = []
        
        for product in initial_recommendations:
            # Get reviews for this product
            product_reviews = self.cleaned_data[
                self.cleaned_data['name'] == product
            ]['reviews_text'].tolist()
            
            if len(product_reviews) == 0:
                continue
            
            # Predict sentiment probabilities
            sentiment_probs = self.predict_sentiment(product_reviews)
            
            # Calculate metrics
            avg_sentiment = np.mean(sentiment_probs)
            positive_ratio = np.mean(sentiment_probs > 0.5)
            
            # Get average rating for this product
            avg_rating = self.cleaned_data[
                self.cleaned_data['name'] == product
            ]['reviews_rating'].mean()
            
            product_sentiments.append({
                'product': product,
                'sentiment_score': avg_sentiment,
                'positive_ratio': positive_ratio,
                'avg_rating': avg_rating,
                'num_reviews': len(product_reviews)
            })
        
        # Sort by sentiment score and return top N
        sorted_products = sorted(
            product_sentiments, 
            key=lambda x: x['sentiment_score'], 
            reverse=True
        )
        
        return sorted_products[:n_final]
    
    def is_valid_user(self, username):
        """
        Check if a username exists in the system.
        
        Args:
            username (str): Username to check
            
        Returns:
            bool: True if user exists, False otherwise
        """
        if self.valid_users is None:
            return False
        return username in self.valid_users
    
    def get_all_users(self):
        """
        Get list of all valid users in the system.
        
        Returns:
            list: List of valid usernames
        """
        return self.valid_users if self.valid_users else []
    
    def get_sample_users(self, n=10):
        """
        Get a sample of valid users for display/testing.
        
        Args:
            n (int): Number of sample users to return
            
        Returns:
            list: List of sample usernames
        """
        if self.valid_users is None:
            return []
        return self.valid_users[:n]


# Singleton instance for the Flask app
recommender = SentimentRecommender()


def initialize_models(model_path='models/'):
    """
    Initialize the recommender system with pre-trained models.
    
    Args:
        model_path (str): Path to the models directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    return recommender.load_models(model_path)


def get_recommendations(username):
    """
    Get sentiment-filtered recommendations for a user.
    
    Args:
        username (str): Username to get recommendations for
        
    Returns:
        list or None: List of recommendations or None if user not found
    """
    return recommender.get_sentiment_filtered_recommendations(username)


def check_user(username):
    """
    Check if a user exists in the system.
    
    Args:
        username (str): Username to check
        
    Returns:
        bool: True if user exists
    """
    return recommender.is_valid_user(username)


def get_sample_users(n=10):
    """
    Get sample users for display.
    
    Args:
        n (int): Number of users to return
        
    Returns:
        list: List of usernames
    """
    return recommender.get_sample_users(n)

