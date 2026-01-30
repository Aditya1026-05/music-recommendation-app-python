# preprocess.py
import os
import pandas as pd
import re
import nltk
import joblib
import logging

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================== PATH SETUP ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "spotify_millsongdata.csv")
DF_PATH = os.path.join(BASE_DIR, "df_cleaned.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
COSINE_PATH = os.path.join(BASE_DIR, "cosine_sim.pkl")

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

# ==================== NLTK DATA ====================
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

# ==================== TEXT CLEANING ====================
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ==================== MAIN FUNCTION ====================
def run_preprocessing():
    logging.info("🚀 Starting preprocessing...")

    # Load dataset
    try:
        df = pd.read_csv(DATASET_PATH)
        df = df.sample(n=10000, random_state=42)
        logging.info("✅ Dataset loaded and sampled: %d rows", len(df))
    except Exception as e:
        logging.error("❌ Failed to load dataset: %s", str(e))
        raise e

    # Drop unnecessary columns
    df = df.drop(columns=["link"], errors="ignore").reset_index(drop=True)

    # Clean text
    logging.info("🧹 Cleaning text...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)
    logging.info("✅ Text cleaned.")

    # TF-IDF
    logging.info("🔠 Vectorizing using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])
    logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

    # Cosine similarity
    logging.info("📐 Calculating cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    logging.info("✅ Cosine similarity matrix generated.")

    # Save files
    joblib.dump(df, DF_PATH)
    joblib.dump(tfidf_matrix, TFIDF_PATH)
    joblib.dump(cosine_sim, COSINE_PATH)

    logging.info("💾 Data saved to disk.")
    logging.info("✅ Preprocessing complete.")

# ==================== RUN DIRECTLY ====================
if __name__ == "__main__":
    run_preprocessing()
