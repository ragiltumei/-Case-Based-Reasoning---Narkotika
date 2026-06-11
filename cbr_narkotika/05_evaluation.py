"""
05_evaluation.py  ─  Tahap 5: Model Evaluation
===============================================
- Evaluasi retrieval: Accuracy, Precision, Recall, F1-score
- Evaluasi prediksi solusi
- Analisis kegagalan (error analysis / rejection analysis)
- Output: data/eval/retrieval_metrics.csv
           data/eval/prediction_metrics.csv
           data/eval/evaluation_report.txt
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score,
                                      classification_report,
                                      confusion_matrix)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection  import cross_val_score, StratifiedKFold
from collections              import Counter

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
EVAL_DIR   = os.path.join(BASE_DIR, "data", "eval")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "data", "results")
os.makedirs(EVAL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_artefacts():
    df = pd.read_csv(os.path.join(PROC_DIR, "cases.csv"), encoding="utf-8-sig")
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
    with open(os.path.join(EVAL_DIR, "queries.json"), "r", encoding="utf-8") as f:
        queries = json.load(f)
    return df, vectorizer, tfidf_matrix, clf, le, queries


def eval_retrieval(queries, df, vectorizer, tfidf_matrix, k=5):
    """
    Evaluate Hit@k retrieval for each query.
    Returns per-query results and aggregate metrics.
    """
    results = []
    for q in queries:
        q_vec = vectorizer.transform([q["query_text"].lower()])
        sims  = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_k = [int(df.iloc[i]["case_id"]) for i in np.argsort(sims)[::-1][:k]]
        gt    = q["ground_truth_ids"]

        hit      = int(any(c in gt for c in top_k))
        precision = len(set(top_k) & set(gt)) / k
        recall    = len(set(top_k) & set(gt)) / max(len(gt), 1)
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results.append({
            "query_id"      : q["query_id"],
            "query_text"    : q["query_text"][:60] + "...",
            "ground_truth"  : gt,
            "top_k"         : top_k,
            "hit_at_k"      : hit,
            "precision_at_k": round(precision, 4),
            "recall_at_k"   : round(recall, 4),
            "f1_at_k"       : round(f1, 4),
        })

    r_df = pd.DataFrame(results)
    agg  = {
        "Hit@5"    : r_df["hit_at_k"].mean(),
        "Precision": r_df["precision_at_k"].mean(),
        "Recall"   : r_df["recall_at_k"].mean(),
        "F1"       : r_df["f1_at_k"].mean(),
    }
    return r_df, agg


def eval_svm_classification(df, vectorizer, tfidf_matrix, le, clf):
    """Full classification report on the corpus using cross-validation."""
    y_true = le.transform(df["label_putusan"].tolist())

    # Cross-validation
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(clf, tfidf_matrix, y_true, cv=cv, scoring="accuracy")

    # Full corpus predictions
    y_pred = clf.predict(tfidf_matrix)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cr   = classification_report(
        y_true, y_pred, target_names=le.classes_, zero_division=0
    )
    cm   = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "cv_accuracy_mean": cv_acc.mean(), "cv_accuracy_std": cv_acc.std(),
        "classification_report": cr, "confusion_matrix": cm,
        "y_true": y_true, "y_pred": y_pred, "labels": le.classes_,
    }


def error_analysis(retrieval_df, clf_result, df, le):
    """Identify failure cases and provide recommendations."""
    errors = []

    # Retrieval misses
    missed = retrieval_df[retrieval_df["hit_at_k"] == 0]
    for _, row in missed.iterrows():
        errors.append({
            "type"      : "Retrieval Miss",
            "query_id"  : row["query_id"],
            "detail"    : f"GT={row['ground_truth']} not in top-5={row['top_k']}",
            "reason"    : "Query terms tidak cukup unik dibanding dokumen lain",
            "recommendation": "Tambah bobot pada field pasal dan jenis_narkoba"
        })

    # Classification errors
    y_t, y_p, labels = clf_result["y_true"], clf_result["y_pred"], clf_result["labels"]
    for i, (yt, yp) in enumerate(zip(y_t, y_p)):
        if yt != yp:
            row = df.iloc[i]
            errors.append({
                "type"       : "Classification Error",
                "query_id"   : f"case_{row['case_id']:03d}",
                "detail"     : f"True={labels[yt]}, Pred={labels[yp]} | {row['terdakwa']}",
                "reason"     : "Imbalanced labels (22 seumur hidup vs 13 penjara)",
                "recommendation": "Gunakan SMOTE atau class_weight='balanced' di SVM"
            })
    return errors


def plot_metrics(agg_retrieval, clf_result, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("CBR Narkotika — Model Evaluation Dashboard", fontsize=14, fontweight="bold")

    # ── Chart 1: Retrieval metrics bar ──
    ax1 = axes[0]
    metrics = list(agg_retrieval.keys())
    values  = [agg_retrieval[m] for m in metrics]
    colors  = ["#2196F3","#4CAF50","#FF9800","#9C27B0"]
    bars = ax1.bar(metrics, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Retrieval Metrics (TF-IDF, k=5)", fontsize=11)
    ax1.set_ylabel("Score")
    ax1.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="Threshold 0.8")
    ax1.legend(fontsize=8)

    # ── Chart 2: SVM Classification metrics ──
    ax2 = axes[1]
    clf_metrics  = ["Accuracy","Precision","Recall","F1","CV Acc."]
    clf_vals     = [clf_result["accuracy"], clf_result["precision"],
                    clf_result["recall"],   clf_result["f1"],
                    clf_result["cv_accuracy_mean"]]
    bars2 = ax2.bar(clf_metrics, clf_vals, color=["#FF5722","#E91E63","#009688","#795548","#607D8B"],
                    edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars2, clf_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.set_title("SVM Classification Metrics", fontsize=11)
    ax2.set_ylabel("Score")

    # ── Chart 3: Confusion Matrix ──
    ax3 = axes[2]
    cm  = clf_result["confusion_matrix"]
    lbs = clf_result["labels"]
    im  = ax3.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax3.set_xticks(range(len(lbs)))
    ax3.set_yticks(range(len(lbs)))
    ax3.set_xticklabels([l.replace(" ", "\n") for l in lbs], fontsize=8)
    ax3.set_yticklabels(lbs, fontsize=8)
    for i in range(len(lbs)):
        for j in range(len(lbs)):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black",
                     fontsize=12, fontweight="bold")
    ax3.set_title("Confusion Matrix (SVM)", fontsize=11)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")
    plt.colorbar(im, ax=ax3, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafik tersimpan → {save_path}")


def write_report(agg_retrieval, clf_result, retrieval_df, errors):
    lines = []
    lines.append("=" * 65)
    lines.append("LAPORAN EVALUASI MODEL CBR")
    lines.append("Domain: Pidana Khusus Narkotika & Psikotropika")
    lines.append("Metode: TF-IDF + LinearSVC")
    lines.append("=" * 65)

    lines.append("\n── RETRIEVAL METRICS (TF-IDF Cosine Similarity, k=5) ──")
    for k, v in agg_retrieval.items():
        lines.append(f"  {k:<12}: {v:.4f}")

    lines.append("\n── PER-QUERY RETRIEVAL DETAIL ──")
    for _, row in retrieval_df.iterrows():
        lines.append(
            f"  {row['query_id']}: Hit={row['hit_at_k']}  "
            f"P={row['precision_at_k']:.3f}  R={row['recall_at_k']:.3f}  "
            f"F1={row['f1_at_k']:.3f}"
        )

    lines.append("\n── SVM CLASSIFICATION METRICS ──")
    lines.append(f"  Accuracy  : {clf_result['accuracy']:.4f}")
    lines.append(f"  Precision : {clf_result['precision']:.4f}")
    lines.append(f"  Recall    : {clf_result['recall']:.4f}")
    lines.append(f"  F1-Score  : {clf_result['f1']:.4f}")
    lines.append(f"  CV Acc.   : {clf_result['cv_accuracy_mean']:.4f} ± {clf_result['cv_accuracy_std']:.4f}")
    lines.append("\n  Classification Report (per-class):")
    for l in clf_result["classification_report"].split("\n"):
        lines.append("    " + l)

    lines.append("\n── ERROR / REJECTION ANALYSIS ──")
    if errors:
        for e in errors:
            lines.append(f"  [{e['type']}] {e['query_id']}")
            lines.append(f"    Detail  : {e['detail']}")
            lines.append(f"    Reason  : {e['reason']}")
            lines.append(f"    Fix     : {e['recommendation']}")
    else:
        lines.append("  Tidak ada error ditemukan.")

    lines.append("\n── REKOMENDASI PERBAIKAN ──")
    lines.append("  1. Data Augmentation: tambah dokumen hingga ≥100 putusan")
    lines.append("     untuk mengurangi bias dataset kecil.")
    lines.append("  2. Class Balancing: gunakan class_weight='balanced'")
    lines.append("     atau SMOTE untuk menangani imbalance label.")
    lines.append("  3. Feature Enrichment: sertakan berat barang bukti")
    lines.append("     sebagai fitur numerik tambahan di SVM.")
    lines.append("  4. Embedding: coba IndoBERT untuk representasi semantik")
    lines.append("     lebih kaya (terutama query pendek).")
    lines.append("  5. Threshold Rejection: jika cosine-sim < 0.15,")
    lines.append("     tandai sebagai 'tidak ada kasus mirip' (rejection).")
    lines.append("=" * 65)

    report_path = os.path.join(EVAL_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Laporan tersimpan → {report_path}")
    return "\n".join(lines)


def main():
    print("=" * 65)
    print("[Tahap 5] Model Evaluation")
    print("=" * 65)

    df, vectorizer, tfidf_matrix, clf, le, queries = load_artefacts()

    # ── Retrieval eval ──
    print("\n[1] Evaluasi Retrieval (Hit@5)...")
    retrieval_df, agg_retrieval = eval_retrieval(queries, df, vectorizer, tfidf_matrix, k=5)
    print(retrieval_df[["query_id","hit_at_k","precision_at_k","recall_at_k","f1_at_k"]].to_string(index=False))
    print(f"\n  Aggregate: {agg_retrieval}")

    # Save retrieval metrics
    rm_path = os.path.join(EVAL_DIR, "retrieval_metrics.csv")
    retrieval_df.to_csv(rm_path, index=False, encoding="utf-8-sig")
    print(f"  Tersimpan → {rm_path}")

    # ── SVM Classification eval ──
    print("\n[2] Evaluasi SVM Classification...")
    clf_result = eval_svm_classification(df, vectorizer, tfidf_matrix, le, clf)
    print(f"  Accuracy  : {clf_result['accuracy']:.4f}")
    print(f"  Precision : {clf_result['precision']:.4f}")
    print(f"  Recall    : {clf_result['recall']:.4f}")
    print(f"  F1        : {clf_result['f1']:.4f}")
    print(f"  CV Acc    : {clf_result['cv_accuracy_mean']:.4f} ± {clf_result['cv_accuracy_std']:.4f}")

    # Save prediction metrics
    pred_df = pd.read_csv(os.path.join(RESULT_DIR, "predictions.csv"), encoding="utf-8-sig")
    pm_path = os.path.join(EVAL_DIR, "prediction_metrics.csv")
    pred_summary = pd.DataFrame([{
        "total_predictions" : len(pred_df),
        "correct"           : pred_df["match"].sum(),
        "accuracy"          : pred_df["match"].mean(),
    }])
    pred_summary.to_csv(pm_path, index=False, encoding="utf-8-sig")
    print(f"\n  Prediction Accuracy: {pred_df['match'].mean():.2%}")
    print(f"  Tersimpan → {pm_path}")

    # ── Error Analysis ──
    print("\n[3] Error Analysis...")
    errors = error_analysis(retrieval_df, clf_result, df, le)
    print(f"  Total errors/rejections: {len(errors)}")
    for e in errors[:3]:
        print(f"  [{e['type']}] {e['query_id']}: {e['detail'][:60]}")

    # ── Plot ──
    print("\n[4] Membuat visualisasi...")
    plot_path = os.path.join(EVAL_DIR, "performance_chart.png")
    plot_metrics(agg_retrieval, clf_result, plot_path)

    # ── Report ──
    print("\n[5] Menulis laporan evaluasi...")
    report = write_report(agg_retrieval, clf_result, retrieval_df, errors)
    print(report[-300:])   # print tail of report

    print("\n=== Tahap 5 Selesai ===")


if __name__ == "__main__":
    main()
