# app.py - Flask app for product recommendations
# Ebuss Capstone Project

from flask import Flask, render_template, request, jsonify
import os
from model import initialize_models, get_recommendations, check_user, get_sample_users

app = Flask(__name__)
models_loaded = False

def load_models():
    global models_loaded
    path = os.environ.get('MODEL_PATH', 'models/')
    models_loaded = initialize_models(path)
    print("Models loaded!" if models_loaded else "Failed to load models!")
    return models_loaded


@app.route('/')
def home():
    users = get_sample_users(20) if models_loaded else []
    return render_template('index.html', sample_users=users, models_loaded=models_loaded)


@app.route('/recommend', methods=['POST'])
def recommend():
    if not models_loaded:
        return jsonify({'success': False, 'error': 'Models not loaded. Please contact the administrator.'})
    
    # get username from request
    if request.is_json:
        username = request.get_json().get('username', '').strip()
    else:
        username = request.form.get('username', '').strip()
    
    if not username:
        return jsonify({'success': False, 'error': 'Please enter a username.'})
    
    if not check_user(username):
        return jsonify({
            'success': False,
            'error': f'User "{username}" not found. Try one of the sample usernames.',
            'sample_users': get_sample_users(5)
        })
    
    # get recs
    try:
        recs = get_recommendations(username)
        
        if recs is None:
            return jsonify({'success': False, 'error': f'User "{username}" not found.'})
        
        if len(recs) == 0:
            return jsonify({
                'success': True, 'username': username, 'recommendations': [],
                'message': f'No recommendations for "{username}" yet.'
            })
        
        # format for frontend
        formatted = []
        for i, r in enumerate(recs, 1):
            formatted.append({
                'rank': i,
                'product': r['product'],
                'sentiment_score': round(r['sentiment_score'] * 100, 1),
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
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})


@app.route('/api/users')
def api_users():
    if not models_loaded:
        return jsonify({'users': []})
    n = request.args.get('n', 20, type=int)
    return jsonify({'users': get_sample_users(min(n, 100))})


@app.route('/api/health')
def health():
    """Health check - also shows model file info for debugging"""
    models_dir = 'models/'
    files = {}
    
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            fpath = os.path.join(models_dir, f)
            sz = os.path.getsize(fpath)
            # check for LFS pointer
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
    return render_template('index.html', error='Page not found',
                          sample_users=get_sample_users(20) if models_loaded else []), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html', error='Server error',
                          sample_users=get_sample_users(20) if models_loaded else []), 500


# load models on startup
with app.app_context():
    load_models()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
