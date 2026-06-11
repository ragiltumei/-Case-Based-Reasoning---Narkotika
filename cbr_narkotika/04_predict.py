"""
04_predict.py  ─  Tahap 4: Case Solution Reuse
===============================================
- Mengambil top-5 kasus mirip → ekstrak amar putusan
- Majority vote + weighted similarity untuk prediksi solusi
- Output: data/results/predictions.csv
"""

import os, json, pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from typing import List

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
EVAL_DIR    = os.path.join(BASE_DIR, "data", "eval")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULT_DIR  = os.path.join(BASE_DIR, "data", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

CSV_PATH        = os.path.join(PROC_DIR, "cases.csv")
QUERIES_PATH    = os.path.join(EVAL_DIR,  "queries.json")


# ── Load artefacts ────────────────────────────────────────────────────────────
def load_artefacts():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["text_repr"] = (
        df["ringkasan_dakwaan"].fillna("") + " " +
        df["ringkasan_fakta"].fillna("")   + " " +
        df["argumen_hukum"].fillna("")     + " " +
        df["pasal"].fillna("")             + " " +
        df["jenis_narkoba"].fillna("")
    )

    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "svm_classifier.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le  = pickle.load(f)
    tfidf_matrix = sp.load_npz(os.path.join(MODEL_DIR, "tfidf_matrix.npz"))

    return df, vectorizer, tfidf_matrix, clf, le


# ── Case solution dictionary ──────────────────────────────────────────────────
def build_solution_dict(df) -> dict:
    """Map case_id → {amar, label, pasal, narkoba}"""
    sol = {}
    for _, row in df.iterrows():
        sol[int(row["case_id"])] = {
            "amar_putusan"  : str(row["amar_putusan"]),
            "label_putusan" : str(row["label_putusan"]),
            "pasal"         : str(row["pasal"]),
            "jenis_narkoba" : str(row["jenis_narkoba"]),
        }
    return sol


# ── retrieve ──────────────────────────────────────────────────────────────────
def retrieve(query: str, vectorizer, tfidf_matrix, df, k: int = 5):
    q_vec = vectorizer.transform([query.lower()])
    sims  = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_k = np.argsort(sims)[::-1][:k]
    case_ids  = [int(df.iloc[i]["case_id"]) for i in top_k]
    sim_vals  = sims[top_k].tolist()
    return case_ids, sim_vals


# ── predict_outcome ───────────────────────────────────────────────────────────
def predict_outcome(
    query: str,
    vectorizer,
    tfidf_matrix,
    df,
    case_solutions: dict,
    k: int = 5,
    method: str = "weighted",          # "majority" or "weighted"
) -> dict:
    """
    1. Retrieve top-k similar cases
    2. Aggregate solutions:
       - majority:  most frequent label
       - weighted:  cosine-weighted vote
    Returns dict with predicted_label, representative_amar, top_k_info
    """
    top_k_ids, sims = retrieve(query, vectorizer, tfidf_matrix, df, k)
    solutions       = [case_solutions[cid] for cid in top_k_ids]

    labels = [s["label_putusan"] for s in solutions]

    if method == "majority":
        predicted_label = Counter(labels).most_common(1)[0][0]
    else:  # weighted
        weights = {lab: 0.0 for lab in set(labels)}
        for lab, sim in zip(labels, sims):
            weights[lab] += sim
        predicted_label = max(weights, key=weights.get)

    # Pick the highest-similarity case with predicted_label as representative
    best_idx  = next(
        (i for i, s in enumerate(solutions) if s["label_putusan"] == predicted_label),
        0
    )
    rep_amar  = solutions[best_idx]["amar_putusan"]
    rep_pasal = solutions[best_idx]["pasal"]

    return {
        "predicted_label"   : predicted_label,
        "representative_amar": rep_amar,
        "top_5_case_ids"    : top_k_ids,
        "top_5_similarities": [round(s, 4) for s in sims],
        "top_5_labels"      : labels,
        "representative_pasal": rep_pasal,
    }


# ── Demo: 5 new case queries ──────────────────────────────────────────────────
NEW_CASES = [
    {
        "query_id"       : "NEW-001",
        "query_text"     : "terdakwa ditemukan membawa sabu-sabu 3 gram dalam kantong celana oleh polisi saat razia",
        "actual_label"   : "pidana penjara",
    },
    {
        "query_id"       : "NEW-002",
        "query_text"     : "tersangka menyimpan 1500 gram heroin dikemas dalam paket di lemari rumahnya",
        "actual_label"   : "penjara seumur hidup",
    },
    {
        "query_id"       : "NEW-003",
        "query_text"     : "terdakwa menanam 10 batang pohon ganja di pekarangan belakang rumah pasal 111",
        "actual_label"   : "pidana penjara",
    },
    {
        "query_id"       : "NEW-004",
        "query_text"     : "permufakatan jahat terdakwa bersama rekannya mengedarkan narkotika pasal 132 dipidana",
        "actual_label"   : "pidana penjara",
    },
    {
        "query_id"       : "NEW-005",
        "query_text"     : "terdakwa memiliki 800 gram kokain golongan I bukan tanaman untuk diedarkan tanpa izin",
        "actual_label"   : "penjara seumur hidup",
    },
]


def main():
    print("=" * 65)
    print("[Tahap 4] Case Solution Reuse")
    print("=" * 65)

    df, vectorizer, tfidf_matrix, clf, le = load_artefacts()
    case_solutions = build_solution_dict(df)

    rows = []
    for nc in NEW_CASES:
        result = predict_outcome(
            nc["query_text"], vectorizer, tfidf_matrix, df, case_solutions,
            k=5, method="weighted"
        )
        match = result["predicted_label"] == nc["actual_label"]

        print(f"\n  [{nc['query_id']}] {nc['query_text'][:60]}...")
        print(f"    Top-5 IDs         : {result['top_5_case_ids']}")
        print(f"    Top-5 Labels      : {result['top_5_labels']}")
        print(f"    Predicted Label   : {result['predicted_label']}")
        print(f"    Actual Label      : {nc['actual_label']}")
        print(f"    Match             : {'✓' if match else '✗'}")
        print(f"    Rep. Amar Putusan : {result['representative_amar'][:80]}...")

        rows.append({
            "query_id"            : nc["query_id"],
            "query_text"          : nc["query_text"],
            "predicted_solution"  : result["predicted_label"],
            "representative_amar" : result["representative_amar"],
            "actual_label"        : nc["actual_label"],
            "match"               : match,
            "top_5_case_ids"      : str(result["top_5_case_ids"]),
            "top_5_similarities"  : str(result["top_5_similarities"]),
        })

    pred_df   = pd.DataFrame(rows)
    pred_path = os.path.join(RESULT_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    correct = pred_df["match"].sum()
    print(f"\n✓ Predictions tersimpan → {pred_path}")
    print(f"  Demo Accuracy: {correct}/{len(pred_df)} = {correct/len(pred_df):.2%}")
    print("=== Tahap 4 Selesai ===")
    return pred_df


if __name__ == "__main__":
    main()
