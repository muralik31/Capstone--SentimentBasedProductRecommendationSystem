# app.py - Flask web app for Ebuss Product Recommendations
# --------------------------------------------------------
# Kept this file minimal on purpose - all the ML logic lives in model.py.
# This just handles HTTP requests and renders templates.

from flask import Flask, render_template, request, jsonify
import os
from model import initialize_models, get_recommendations, check_user, get_sample_users

app = Flask(__name__)
models_loaded = False


def load_models():
    """
    Load ML models at startup.
    
    Using environment variable for path because Hugging Face Spaces
    has different directory structure than local dev. Defaults to 'models/'
    which works locally and on most deployment platforms.
    """
    global models_loaded
    path = os.environ.get('MODEL_PATH', 'models/')
    models_loaded = initialize_models(path)
    print("Models loaded!" if models_loaded else "Failed to load models!")
    return models_loaded


@app.route('/')
def home():
    """
    Render the main page.
    
    Passing sample users to template so users can click to try them -
    makes it easier for evaluators to test without guessing valid usernames.
    """
    users = get_sample_users(20) if models_loaded else []
    return render_template('index.html', sample_users=users, models_loaded=models_loaded)


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Main recommendation endpoint - called when user submits the form.
    
    Flow:
    1. Validate username exists in our dataset
    2. Get top 5 sentiment-filtered recommendations
    3. Format nicely for the frontend (percentages, rounded numbers)
    
    Returns JSON because the frontend uses fetch() - cleaner than page reloads.
    """
    if not models_loaded:
        return jsonify({
            'success': False, 
            'error': 'Models not loaded. Please contact the administrator.'
        })
    
    # Handle both JSON and form data (JSON from fetch, form from direct POST)
    if request.is_json:
        username = request.get_json().get('username', '').strip()
    else:
        username = request.form.get('username', '').strip()
    
    if not username:
        return jsonify({'success': False, 'error': 'Please enter a username.'})
    
    # Check if user exists before doing expensive recommendation computation
    if not check_user(username):
        return jsonify({
            'success': False,
            'error': f'User "{username}" not found. Try one of the sample usernames.',
            'sample_users': get_sample_users(5)
        })
    
    try:
        recs = get_recommendations(username)
        
        if recs is None:
            return jsonify({'success': False, 'error': f'User "{username}" not found.'})
        
        if len(recs) == 0:
            # User exists but has rated everything or CF couldn't find matches
            return jsonify({
                'success': True, 'username': username, 'recommendations': [],
                'message': f'No new recommendations for "{username}" yet.'
            })
        
        # Format for frontend display - convert decimals to percentages
        formatted = []
        for i, r in enumerate(recs, 1):
            formatted.append({
                'rank': i,
                'product': r['product'],
                'sentiment_score': round(r['sentiment_score'] * 100, 1),  # 0.85 -> 85.0%
                'positive_ratio': round(r['positive_ratio'] * 100, 1),
                'avg_rating': round(r['avg_rating'], 1),
                'num_reviews': r['num_reviews']
            })
        
        return jsonify({
            'success': True,
            'username': username,
            'recommendations': formatted,
            'message': f'Top 5 recommendations for {username}'
        })
        
    except Exception as e:
        # Log full error server-side but send generic message to user
        print(f"Recommendation error for {username}: {e}")
        return jsonify({'success': False, 'error': f'Error generating recommendations. Please try again.'})


@app.route('/api/users')
def api_users():
    """
    Get list of valid usernames (for autocomplete or testing).
    Capped at 100 to avoid sending huge response.
    """
    if not models_loaded:
        return jsonify({'users': []})
    n = request.args.get('n', 20, type=int)
    return jsonify({'users': get_sample_users(min(n, 100))})


@app.route('/api/health')
def health():
    """
    Health check endpoint.
    
    Added detailed model file info after Render deployment issues -
    helps diagnose whether models are actual files or just LFS pointers.
    Not strictly necessary for production but saved me hours of debugging.
    """
    models_dir = 'models/'
    files = {}
    
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            fpath = os.path.join(models_dir, f)
            sz = os.path.getsize(fpath)
            is_lfs = False
            if sz < 200:
                try:
                    with open(fpath, 'r') as fp:
                        is_lfs = fp.read(30).startswith('version https://git-lfs')
                except: pass
            files[f] = {'size_mb': round(sz/(1024*1024), 2), 'is_lfs_pointer': is_lfs}
    
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'model_files': files
    })


@app.errorhandler(404)
def not_found(e):
    """Custom 404 - show main page with error message instead of ugly default"""
    return render_template('index.html', error='Page not found',
                          sample_users=get_sample_users(20) if models_loaded else []), 404

@app.errorhandler(500)
def server_error(e):
    """Custom 500 - at least show something useful"""
    return render_template('index.html', error='Server error',
                          sample_users=get_sample_users(20) if models_loaded else []), 500


# Load models when app starts (not on each request)
# Using app_context() because Flask needs context for some operations
with app.app_context():
    load_models()


if __name__ == '__main__':
    # Port 7860 is Hugging Face Spaces default, 5000 is Flask default
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
