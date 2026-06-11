#  CBR — Sistem Analisis Putusan Pengadilan Narkotika

> **Case-Based Reasoning for Indonesian Narcotics Court Decision Analysis**

|                   |                                                                |
|---                |---                                                             |
| **Mata Kuliah**   | Penalaran Komputer                                             |
| **SubCPMK**       | SubCPMK-3 — Siklus Case-Based Reasoning                        |
| **Domain**        | Pidana Khusus Narkotika & Psikotropika (UU No. 35 Tahun 2009)  |
| **Universitas**   | Universitas Muhammadiyah Malang — Teknik Informatika           |
| **Semester**      | Genap 2025/2026                                                |
| **Metode**        | TF-IDF + LinearSVM                                             |

---

##  Deskripsi Proyek

Sistem **Case-Based Reasoning (CBR)** berbasis Python untuk mendukung analisis putusan pengadilan **Pidana Khusus Narkotika & Psikotropika**. Data putusan bersumber dari Direktori Putusan Mahkamah Agung Republik Indonesia.

Sistem mengimplementasikan seluruh siklus CBR:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Case Base│ →  │   Case   │ →  │   Case   │ →  │Solution  │ →  │ Model    │
│ Building │    │Represent.│    │ Retrieval│    │  Reuse   │    │ Evaluation│
│(Tahap 1) │    │(Tahap 2) │    │(Tahap 3) │    │(Tahap 4) │    │(Tahap 5) │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

##  Struktur Repository

```
cbr_narkotika/
│
├── data/                          # Raw & processed data
│   ├── raw/                       # Dokumen putusan (.txt) — 35 file
│   │   ├── case_001.txt
│   │   ├── case_002.txt
│   │   └── ...
│   ├── processed/                 # Data terstruktur
│   │   ├── cases.csv              # Dataset utama (35 baris, 17 kolom)
│   │   ├── cases.json             # Dataset format JSON
│   │   └── cases_meta.json        # Metadata putusan
│   ├── eval/                      # Hasil evaluasi
│   │   ├── queries.json           # 7 query uji + ground-truth
│   │   ├── retrieval_metrics.csv  # Metrik retrieval per query
│   │   ├── prediction_metrics.csv # Metrik klasifikasi
│   │   ├── evaluation_report.txt  # Laporan evaluasi lengkap
│   │   └── performance_chart.png  # Grafik performa model
│   └── results/
│       └── predictions.csv        # Hasil prediksi 5 kasus baru
│
├── notebooks/                     # Jupyter Notebooks per tahap
│   ├── Tahap1_Case_Base.ipynb
│   ├── Tahap2_Case_Representation.ipynb
│   ├── Tahap3_Case_Retrieval.ipynb
│   ├── Tahap4_Solution_Reuse.ipynb
│   └── Tahap5_Model_Evaluation.ipynb
│
├── models/                        # Trained models (auto-generated)
│   ├── tfidf_vectorizer.pkl
│   ├── svm_classifier.pkl
│   ├── label_encoder.pkl
│   └── tfidf_matrix.npz
│
├── logs/
│   └── cleaning.log               # Log preprocessing dokumen
│
├── generate_data.py               # Generator data (ganti dengan scraper MA RI)
├── 01_case_base.py                # Tahap 1: Preprocessing & Cleaning
├── 02_case_representation.py      # Tahap 2: Metadata & Feature Engineering
├── 03_retrieval.py                # Tahap 3: TF-IDF + SVM Retrieval
├── 04_predict.py                  # Tahap 4: Solution Reuse
├── 05_evaluation.py               # Tahap 5: Model Evaluation
├── requirements.txt
└── README.md
```

---

##  Instalasi

### Prasyarat
- Python **3.9+**
- pip

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/ragiltumei/-Case-Based-Reasoning---Narkotika.git
cd cbr-narkotika

# 2. (Opsional tapi disarankan) Buat virtual environment
python -m venv venv

# Aktivasi — Linux/macOS:
source venv/bin/activate
# Aktivasi — Windows:
venv\Scripts\activate

# 3. Install semua dependensi
pip install -r requirements.txt
```

### Verifikasi Instalasi

```bash
python -c "import pandas, sklearn, numpy, matplotlib, scipy; print('✓ Semua paket berhasil diinstall')"
```

---

##  Cara Menjalankan Pipeline

### Opsi A — Jupyter Notebook *(Direkomendasikan)*

Jalankan notebook **secara berurutan** dari Tahap 1 hingga Tahap 5:

```bash
# Jalankan Jupyter
jupyter notebook

# Buka notebook sesuai urutan:
# notebooks/Tahap1_Case_Base.ipynb
# notebooks/Tahap2_Case_Representation.ipynb
# notebooks/Tahap3_Case_Retrieval.ipynb
# notebooks/Tahap4_Solution_Reuse.ipynb
# notebooks/Tahap5_Model_Evaluation.ipynb
```

### Opsi B — Script Python End-to-End

```bash
# Tahap 0: Generate/siapkan data dokumen putusan
python generate_data.py

# Tahap 1: Bangun Case Base (cleaning & validasi)
python 01_case_base.py

# Tahap 2: Case Representation (ekstraksi metadata & fitur)
python 02_case_representation.py

# Tahap 3: Case Retrieval (TF-IDF vectorization + SVM training)
python 03_retrieval.py

# Tahap 4: Case Solution Reuse (prediksi solusi kasus baru)
python 04_predict.py

# Tahap 5: Model Evaluation (metrics, visualisasi, error analysis)
python 05_evaluation.py
```

---

##  Contoh Penggunaan (Live Retrieval)

```python
import pickle, scipy.sparse as sp, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load artefakt model
vectorizer   = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
tfidf_matrix = sp.load_npz('models/tfidf_matrix.npz')
df           = pd.read_csv('data/processed/cases.csv', encoding='utf-8-sig')

def retrieve(query: str, k: int = 5):
    """
    Temukan top-k kasus paling mirip dengan query.
    
    Parameters
    ----------
    query : str  — Deskripsi kasus baru (teks bebas bahasa Indonesia)
    k     : int  — Jumlah kasus yang dikembalikan (default 5)
    
    Returns
    -------
    list — [(case_id, similarity_score), ...]
    """
    q_vec = vectorizer.transform([query.lower()])
    sims  = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_k = np.argsort(sims)[::-1][:k]
    return [(int(df.iloc[i]['case_id']), round(float(sims[i]), 4)) for i in top_k]


# Contoh query kasus baru
query = "terdakwa ditemukan memiliki sabu-sabu seberat 50 gram tanpa izin, pasal 112 narkotika golongan I"

print(f'Query: "{query}"\n')
print(f'{"Rank":<5} {"Case_ID":<9} {"Similarity":<12} {"Terdakwa":<22} {"Narkoba":<14} {"Label Putusan"}')
print('-' * 85)
for rank, (case_id, sim) in enumerate(retrieve(query, k=5), 1):
    row = df[df['case_id'] == case_id].iloc[0]
    print(f'{rank:<5} {case_id:<9} {sim:<12.4f} {row["terdakwa"]:<22} {row["jenis_narkoba"]:<14} {row["label_putusan"]}')
```

**Output contoh:**
```
Query: "terdakwa ditemukan memiliki sabu-sabu seberat 50 gram tanpa izin, pasal 112 narkotika golongan I"

Rank  Case_ID   Similarity   Terdakwa               Narkoba        Label Putusan
-------------------------------------------------------------------------------------
1     22        0.3304       Vicky Andrianto        sabu-sabu      penjara seumur hidup
2     30        0.2871       Dani Saputra           sabu-sabu      pidana penjara
3     34        0.2843       Hadi Wijaya            sabu-sabu      pidana penjara
4     10        0.2830       Joko Susilo            sabu-sabu      penjara seumur hidup
5     28        0.2817       Bambang Riyanto        sabu-sabu      penjara seumur hidup
```

---

##  Hasil Evaluasi

| Metrik                | Nilai         | Keterangan                                            |
|--------               |-------        |------------                                           |
| **Hit@5**             | **1.000**     | 7/7 query berhasil menemukan kasus relevan            |
| Precision@5           | 0.743         | Rata-rata 3.7 dari 5 hasil yang dikembalikan relevan  |
| Recall@5              | 0.807         | Rata-rata 80% ground-truth berhasil ditemukan         |
| F1@5                  | 0.760         | Harmonic mean Precision & Recall                      |
| **SVM Accuracy**      | **0.857**     | 30/35 kasus diklasifikasi benar                       |
| SVM Precision (w)     | 0.856         | Weighted average                                      |
| SVM Recall (w)        | 0.857         | Weighted average                                      |
| SVM F1 (w)            | 0.856         | Weighted F1-score                                     |
| CV Accuracy (5-fold)  | 0.686 ± 0.167 | Indikator kebutuhan data lebih banyak                 |

---

##  Sumber Data

Dalam implementasi nyata, dokumen putusan diunduh dari:

> **Direktori Putusan Mahkamah Agung RI**  
> 🔗 https://putusan3.mahkamahagung.go.id/  
> Filter: **Klasifikasi → Pidana Khusus → Narkotika & Psikotropika**

Pada repositori ini, `generate_data.py` menghasilkan **35 dokumen sintetis** yang merepresentasikan struktur asli putusan MA RI (nomor perkara, dakwaan, fakta hukum, analisis, amar putusan).

---

##  Tech Stack

| Komponen              | Library / Version                 |
|----------             |-------------------                |
| Bahasa                | Python 3.9+                       |
| Data Processing       | pandas ≥ 2.0, numpy ≥ 1.24        |
| TF-IDF Vectorization  | scikit-learn ≥ 1.3                |      
| ML Model (SVM)        | scikit-learn (LinearSVC)          |
| Similarity Search     | scikit-learn (cosine_similarity)  |
| Model Persistence     | pickle, scipy.sparse              |
| Visualization         | matplotlib ≥ 3.7                  |
| Text Extraction       | pdfminer.six, beautifulsoup4      |
| Notebook              | jupyter, ipykernel                |

---

##  Anggota Tim

| Nama                      | NIM               |
|------                     |-----              |
| [Erdy Muhammad Fakhri]    | [202310370311071] |
| [Ragil Tumei Wijayanto]   | [202310370311240] |

[Penalaran Komputer D ]

---

##  Lisensi

Proyek ini dibuat untuk keperluan tugas akademik.  
Data putusan pengadilan bersumber dari domain publik (Direktori MA RI).

