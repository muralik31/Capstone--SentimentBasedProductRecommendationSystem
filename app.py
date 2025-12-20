"""
Sentiment-Based Product Recommendation System
Flask Application

This Flask application serves the sentiment-based product recommendation system.
Users can enter their username and receive personalized product recommendations
that are filtered based on sentiment analysis of product reviews.

Author: Machine Learning Engineer, Ebuss
Date: December 2024
"""

from flask import Flask, render_template, request, jsonify
import os
from model import initialize_models, get_recommendations, check_user, get_sample_users

# Initialize Flask app
app = Flask(__name__)

# Initialize models on startup
models_loaded = False


def load_models():
    """Load all ML models on startup."""
    global models_loaded
    model_path = os.environ.get('MODEL_PATH', 'models/')
    models_loaded = initialize_models(model_path)
    if models_loaded:
        print(" All models loaded successfully!")
    else:
        print(" Failed to load models!")
    return models_loaded


@app.route('/')
def home():
    """Render the home page."""
    sample_users = get_sample_users(20) if models_loaded else []
    return render_template('index.html', 
                         sample_users=sample_users,
                         models_loaded=models_loaded)


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handle recommendation requests.
    
    Accepts a username and returns top 5 sentiment-filtered product recommendations.
    """
    if not models_loaded:
        return jsonify({
            'success': False,
            'error': 'Models not loaded. Please contact the administrator.'
        })
    
    # Get username from form or JSON
    if request.is_json:
        data = request.get_json()
        username = data.get('username', '').strip()
    else:
        username = request.form.get('username', '').strip()
    
    # Validate username
    if not username:
        return jsonify({
            'success': False,
            'error': 'Please enter a username.'
        })
    
    # Check if user exists
    if not check_user(username):
        return jsonify({
            'success': False,
            'error': f'User "{username}" not found in our system. Please try one of the sample usernames below, or check your spelling.',
            'sample_users': get_sample_users(5)
        })
    
    # Get recommendations
    try:
        recommendations = get_recommendations(username)
        
        if recommendations is None:
            return jsonify({
                'success': False,
                'error': f'User "{username}" not found.',
                'sample_users': get_sample_users(5)
            })
        
        if len(recommendations) == 0:
            return jsonify({
                'success': True,
                'username': username,
                'recommendations': [],
                'message': f'No recommendations available for "{username}" at the moment. This user may need more product reviews to generate recommendations.'
            })
        
        # Format recommendations for response
        formatted_recs = []
        for i, rec in enumerate(recommendations, 1):
            formatted_recs.append({
                'rank': i,
                'product': rec['product'],
                'sentiment_score': round(rec['sentiment_score'] * 100, 1),
                'positive_ratio': round(rec['positive_ratio'] * 100, 1),
                'avg_rating': round(rec['avg_rating'], 1),
                'num_reviews': rec['num_reviews']
            })
        
        return jsonify({
            'success': True,
            'username': username,
            'recommendations': formatted_recs,
            'message': f'Top 5 recommendations for {username}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        })


@app.route('/api/users')
def get_users():
    """API endpoint to get sample users."""
    if not models_loaded:
        return jsonify({'users': []})
    
    n = request.args.get('n', 20, type=int)
    users = get_sample_users(min(n, 100))
    return jsonify({'users': users})


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    import os
    models_dir = 'models/'
    model_files = {}
    
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            filepath = os.path.join(models_dir, f)
            size = os.path.getsize(filepath)
            # Check if it's a Git LFS pointer (small file with specific content)
            is_lfs_pointer = False
            if size < 200:
                try:
                    with open(filepath, 'r') as file:
                        content = file.read(50)
                        is_lfs_pointer = content.startswith('version https://git-lfs')
                except:
                    pass
            model_files[f] = {
                'size_bytes': size,
                'size_mb': round(size / (1024*1024), 2),
                'is_lfs_pointer': is_lfs_pointer
            }
    
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'models_directory_exists': os.path.exists(models_dir),
        'model_files': model_files
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('index.html', 
                         error='Page not found',
                         sample_users=get_sample_users(20) if models_loaded else []), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return render_template('index.html', 
                         error='Internal server error. Please try again later.',
                         sample_users=get_sample_users(20) if models_loaded else []), 500


# Load models when the app starts
with app.app_context():
    load_models()


if __name__ == '__main__':
    # Get port from environment variable (for Heroku) or use 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)

