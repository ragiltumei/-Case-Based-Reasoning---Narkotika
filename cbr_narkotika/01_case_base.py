"""
01_case_base.py  ─  Tahap 1: Membangun Case Base
================================================
- Membaca dokumen putusan dari data/raw/
- Membersihkan teks (hapus header/footer ganda, normalisasi, tokenisasi)
- Memvalidasi kelengkapan teks (≥ 80% konten)
- Menyimpan hasil di data/raw/ (file sudah ada) dan log di logs/cleaning.log
"""

import os, re, logging
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
LOG_FILE    = os.path.join(LOGS_DIR, "cleaning.log")

os.makedirs(LOGS_DIR, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)
log = logging.getLogger(__name__)


# ── Preprocessing functions ──────────────────────────────────────────────────
REQUIRED_SECTIONS = ["DAKWAAN", "FAKTA", "MENGADILI"]

def remove_watermark(text: str) -> str:
    """Remove common PDF watermark patterns."""
    text = re.sub(r"(?im)^(SALINAN|TURUNAN|COPY|MAHKAMAH AGUNG RI)\s*$", "", text)
    return text

def remove_page_numbers(text: str) -> str:
    """Remove standalone page-number lines: e.g. '- 5 -', '5', 'Hal 5 dari 20'."""
    text = re.sub(r"(?m)^\s*[-–]\s*\d+\s*[-–]\s*$", "", text)
    text = re.sub(r"(?m)^\s*Hal(?:aman)?\s+\d+\s+dari\s+\d+\s*$", "", text, flags=re.IGNORECASE)
    return text

def normalize_whitespace(text: str) -> str:
    """Collapse multiple blank lines, strip trailing spaces."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def to_lower(text: str) -> str:
    return text.lower()

def tokenize_words(text: str):
    """Simple whitespace + punctuation tokenizer."""
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return tokens

def clean_document(raw_text: str) -> str:
    text = remove_watermark(raw_text)
    text = remove_page_numbers(text)
    text = normalize_whitespace(text)
    return text


def validate_document(cleaned_text: str, raw_text: str) -> dict:
    """
    Returns a dict with:
      valid   (bool)   : True jika lolos semua validasi
      pct     (float)  : persentase konten tersisa vs asli
      missing (list)   : seksi wajib yang hilang
    """
    raw_len     = max(len(raw_text), 1)
    clean_len   = len(cleaned_text)
    pct         = clean_len / raw_len * 100
    missing     = [s for s in REQUIRED_SECTIONS
                   if s not in cleaned_text.upper()]
    valid       = (pct >= 80) and (len(missing) == 0)
    return {"valid": valid, "pct": round(pct, 1), "missing": missing}


# ── Main pipeline ────────────────────────────────────────────────────────────
def build_case_base():
    files   = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".txt"))
    total   = len(files)
    passed  = 0
    failed  = []

    log.info("=" * 60)
    log.info(f"[Tahap 1] Membangun Case Base  |  {datetime.now()}")
    log.info(f"Jumlah file ditemukan: {total}")
    log.info("=" * 60)

    for fname in files:
        fpath = os.path.join(RAW_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()

        cleaned  = clean_document(raw)
        tokens   = tokenize_words(cleaned)
        result   = validate_document(cleaned, raw)

        status = "OK" if result["valid"] else "WARN"
        log.info(
            f"[{status}] {fname:20s} | pct={result['pct']:5.1f}%"
            f" | tokens={len(tokens):5d}"
            + (f" | missing={result['missing']}" if result["missing"] else "")
        )

        # Overwrite with cleaned version
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(cleaned)

        if result["valid"]:
            passed += 1
        else:
            failed.append(fname)

    log.info("-" * 60)
    log.info(f"✓ Lolos validasi : {passed}/{total}")
    if failed:
        log.warning(f"✗ Gagal validasi : {failed}")
    log.info(f"Cleaning log disimpan → {LOG_FILE}")
    return passed, total


if __name__ == "__main__":
    p, t = build_case_base()
    print(f"\n=== Tahap 1 Selesai: {p}/{t} dokumen valid ===")
