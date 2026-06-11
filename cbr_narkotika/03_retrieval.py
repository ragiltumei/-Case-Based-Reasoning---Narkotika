"""
03_retrieval.py  ─  Tahap 3: Case Retrieval
============================================
Pendekatan: TF-IDF + SVM  (Pendekatan 1: Statistik + ML)
- Membangun vektor TF-IDF dari corpus
- Melatih SVM classifier pada label putusan
- Fungsi retrieve(query, k) → top-k case_id via cosine similarity
- Menyimpan query uji + ground-truth di data/eval/queries.json
"""

import os, json, pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text    import TfidfVectorizer
from sklearn.svm                        import SVC, LinearSVC
from sklearn.model_selection            import train_test_split
from sklearn.metrics.pairwise           import cosine_similarity
from sklearn.preprocessing              import LabelEncoder
from typing                             import List

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
EVAL_DIR  = os.path.join(BASE_DIR, "data", "eval")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(EVAL_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = os.path.join(PROC_DIR, "cases.csv")

# ── Stopwords Bahasa Indonesia (minimal set) ─────────────────────────────────
STOPWORDS_ID = set("""
yang dan di ke dari untuk dengan pada adalah telah bahwa atau juga
ini itu tidak ada dalam oleh karena sebagai telah akan dapat lebih
serta para pihak juga tersebut oleh majelis hakim pengadilan negeri
""".split())


def load_data():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["text_repr"] = (
        df["ringkasan_dakwaan"].fillna("") + " " +
        df["ringkasan_fakta"].fillna("") + " " +
        df["argumen_hukum"].fillna("") + " " +
        df["pasal"].fillna("") + " " +
        df["jenis_narkoba"].fillna("")
    )
    return df


def build_tfidf(df) -> tuple:
    """Fit TF-IDF on full corpus, return vectorizer + matrix."""
    vectorizer = TfidfVectorizer(
        max_features   = 3000,
        ngram_range    = (1, 2),
        min_df         = 1,
        sublinear_tf   = True,
        stop_words     = list(STOPWORDS_ID),
    )
    tfidf_matrix = vectorizer.fit_transform(df["text_repr"].tolist())
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


def train_svm(tfidf_matrix, labels) -> tuple:
    """Train LinearSVC on TF-IDF features."""
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc  = clf.score(X_test, y_test)
    print(f"  SVM  Train Accuracy: {train_acc:.4f}")
    print(f"  SVM  Test  Accuracy: {test_acc:.4f}")

    split_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape" : X_test.shape,
        "y_train"      : y_train.tolist(),
        "y_test"       : y_test.tolist(),
        "train_indices": list(range(len(y_train))),
        "test_acc"     : round(test_acc, 4),
    }
    return clf, le, split_info


# ── retrieve() ───────────────────────────────────────────────────────────────
class CBRRetriever:
    def __init__(self, df, vectorizer, tfidf_matrix, clf, le):
        self.df           = df
        self.vectorizer   = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.clf          = clf
        self.le           = le

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """
        1) Pre-process query (basic lowercasing)
        2) Hitung vektor TF-IDF query
        3) Hitung cosine-similarity dengan semua case vectors
        4) Kembalikan top-k case_id (1-indexed)
        """
        # 1) Pre-process
        query_clean = query.lower()
        # 2) Vectorize
        q_vec = self.vectorizer.transform([query_clean])
        # 3) Cosine similarity
        sims  = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        # 4) Top-k
        top_k_idx  = np.argsort(sims)[::-1][:k]
        top_k_cids = [int(self.df.iloc[i]["case_id"]) for i in top_k_idx]
        return top_k_cids, sims[top_k_idx].tolist()

    def predict_label(self, query: str) -> str:
        q_vec = self.vectorizer.transform([query.lower()])
        pred  = self.clf.predict(q_vec)
        return self.le.inverse_transform(pred)[0]


# ── Test queries (ground truth) ──────────────────────────────────────────────
TEST_QUERIES = [
    {
        "query_id"        : "Q001",
        "query_text"      : "terdakwa memiliki sabu-sabu seberat 50 gram tanpa izin disita polisi",
        "expected_label"  : "pidana penjara",
        "ground_truth_ids": [1, 30, 34],
    },
    {
        "query_id"        : "Q002",
        "query_text"      : "terdakwa menyimpan ganja seberat 2 kilogram di rumah pasal 112 narkotika golongan I",
        "expected_label"  : "penjara seumur hidup",
        "ground_truth_ids": [2, 11, 14, 25, 26],
    },
    {
        "query_id"        : "Q003",
        "query_text"      : "terdakwa menjadi perantara jual beli heroin diancam pasal 114 narkotika",
        "expected_label"  : "pidana penjara",
        "ground_truth_ids": [15, 27, 29],
    },
    {
        "query_id"        : "Q004",
        "query_text"      : "terdakwa menguasai ekstasi MDMA lebih dari 5 gram bukan tanaman pasal 112 ayat 2",
        "expected_label"  : "penjara seumur hidup",
        "ground_truth_ids": [18, 19, 21, 24],
    },
    {
        "query_id"        : "Q005",
        "query_text"      : "terdakwa memiliki kokain bukan tanaman tanpa hak barang bukti disita kepolisian",
        "expected_label"  : "penjara seumur hidup",
        "ground_truth_ids": [5, 6, 12, 20, 33],
    },
    {
        "query_id"        : "Q006",
        "query_text"      : "permufakatan jahat jual beli narkotika pasal 132 bersama-sama",
        "expected_label"  : "pidana penjara",
        "ground_truth_ids": [4, 11, 12, 16, 19],
    },
    {
        "query_id"        : "Q007",
        "query_text"      : "terdakwa menanam ganja tanaman narkotika golongan I pasal 111",
        "expected_label"  : "pidana penjara",
        "ground_truth_ids": [5, 14, 17, 27, 30, 32, 34],
    },
]


def main():
    print("=" * 60)
    print("[Tahap 3] Case Retrieval  |  TF-IDF + SVM")
    print("=" * 60)

    # 1. Load data
    df = load_data()
    print(f"\n[1] Data loaded: {len(df)} kasus")

    # 2. Build TF-IDF
    print("\n[2] Building TF-IDF vectors...")
    vectorizer, tfidf_matrix = build_tfidf(df)

    # 3. Train SVM
    print("\n[3] Training SVM classifier (70/30 split)...")
    labels = df["label_putusan"].tolist()
    clf, le, split_info = train_svm(tfidf_matrix, labels)

    # 4. Build retriever
    retriever = CBRRetriever(df, vectorizer, tfidf_matrix, clf, le)

    # 5. Save models
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "svm_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    import scipy.sparse as sp
    sp.save_npz(os.path.join(MODEL_DIR, "tfidf_matrix.npz"), tfidf_matrix)
    print(f"\n[4] Model tersimpan → models/")

    # 6. Run test queries
    print("\n[5] Pengujian Awal — Test Queries:")
    print("-" * 60)
    eval_results = []
    for q in TEST_QUERIES:
        top_k, sims = retriever.retrieve(q["query_text"], k=5)
        pred_label  = retriever.predict_label(q["query_text"])
        hit         = any(c in q["ground_truth_ids"] for c in top_k)
        print(f"  {q['query_id']}: {q['query_text'][:50]}...")
        print(f"    Top-5 retrieved : {top_k}")
        print(f"    Similarities    : {[f'{s:.3f}' for s in sims]}")
        print(f"    Predicted label : {pred_label} (expected: {q['expected_label']})")
        print(f"    Hit@5           : {'✓' if hit else '✗'}")
        eval_results.append({**q, "top_k_retrieved": top_k, "similarities": sims, "predicted_label": pred_label})

    # 7. Save queries.json
    queries_path = os.path.join(EVAL_DIR, "queries.json")
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Queries tersimpan → {queries_path}")
    print("=== Tahap 3 Selesai ===")

    return retriever, df


if __name__ == "__main__":
    main()
