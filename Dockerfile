FROM python:3.11-slim

WORKDIR /app

# Install git-lfs for large files
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# Copy application files
COPY . .

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run the app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]

