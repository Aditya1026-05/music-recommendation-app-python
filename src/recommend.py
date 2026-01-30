import os
import joblib
import logging
from preprocess import run_preprocessing

# -------------------- PATH SETUP --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df_cleaned.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
COSINE_PATH = os.path.join(BASE_DIR, "cosine_sim.pkl")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -------------------- LOAD / GENERATE DATA --------------------
logging.info("📦 Loading data...")

if not (os.path.exists(DF_PATH) and os.path.exists(TFIDF_PATH) and os.path.exists(COSINE_PATH)):
    logging.warning("⚠️ Preprocessed files not found. Running preprocessing...")
    run_preprocessing()

df = joblib.load(DF_PATH)
cosine_sim = joblib.load(COSINE_PATH)

logging.info("✅ Data loaded successfully.")

# -------------------- RECOMMENDER FUNCTION --------------------
def recommend_songs(song_name, top_n=5):
    matches = df[df["song"].str.lower() == song_name.lower()]

    if matches.empty:
        logging.warning("❌ Song not found.")
        return None

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

    song_indices = [i[0] for i in sim_scores]
    result_df = df.loc[song_indices, ["artist", "song"]].reset_index(drop=True)
    result_df.index += 1
    result_df.index.name = "S.No."

    return result_df
