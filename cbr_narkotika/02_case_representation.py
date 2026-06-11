"""
02_case_representation.py  ─  Tahap 2: Case Representation
===========================================================
- Ekstraksi metadata: no_perkara, tanggal, jenis perkara, pasal,
  terdakwa, amar putusan, dll.
- Ekstraksi konten kunci: ringkasan fakta, argumen hukum utama
- Feature engineering: word count, bag-of-words top terms, TF-IDF
- Output: data/processed/cases.csv  &  data/processed/cases.json
"""

import os, re, json
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

META_JSON = os.path.join(PROC_DIR, "cases_meta.json")


# ── Pattern library ───────────────────────────────────────────────────────────
P_NO_PERKARA  = re.compile(r"Nomor\s+([\d/\w.]+)", re.IGNORECASE)
P_TANGGAL     = re.compile(
    r"(\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|"
    r"September|Oktober|November|Desember)\s+\d{4})", re.IGNORECASE)
P_TERDAKWA    = re.compile(r"Nama Lengkap\s*:\s*(.+)", re.IGNORECASE)
P_PASAL       = re.compile(
    r"Pasal\s+(\d+\s+ayat\s*\(\d+\)\s+UU[^\.]+Narkotika)", re.IGNORECASE)
P_JENIS_NARK  = re.compile(
    r"(sabu-sabu|heroin|ganja|ekstasi|kokain|metamfetamina|cannabis|mdma|cocaine)",
    re.IGNORECASE)
P_BERAT       = re.compile(
    r"(\d+(?:[.,]\d+)?\s*(?:gram|kilogram|kg|gr|paket\s*kecil|paket))",
    re.IGNORECASE)
P_AMAR        = re.compile(
    r"(?:Menjatuhkan pidana|pidana penjara|pidana mati|penjara seumur hidup)"
    r"[^\.]{5,120}", re.IGNORECASE)
P_DAKWAAN     = re.compile(
    r"DAKWAAN:(.+?)(?=FAKTA|ANALISIS|MENGADILI)", re.IGNORECASE | re.DOTALL)
P_FAKTA       = re.compile(
    r"FAKTA-FAKTA HUKUM:(.+?)(?=ANALISIS|MENGADILI)", re.IGNORECASE | re.DOTALL)
P_ANALISIS    = re.compile(
    r"ANALISIS HUKUM:(.+?)(?=Hal-hal yang memberatkan|MENGADILI)",
    re.IGNORECASE | re.DOTALL)

BULAN_MAP = {
    "januari":"01","februari":"02","maret":"03","april":"04",
    "mei":"05","juni":"06","juli":"07","agustus":"08",
    "september":"09","oktober":"10","november":"11","desember":"12",
}

def parse_date(raw: str) -> str:
    """Convert '12 Januari 2023' → '2023-01-12'."""
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", raw.strip(), re.IGNORECASE)
    if not m: return raw
    d, mon, y = m.groups()
    return f"{y}-{BULAN_MAP.get(mon.lower(),'??')}-{int(d):02d}"

def extract_section(text: str, pattern) -> str:
    m = pattern.search(text)
    if m:
        raw = m.group(1).strip()
        raw = re.sub(r"\s+", " ", raw)
        return raw[:800]          # cap length
    return ""

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def extract_record(case_id: int, fname: str, text: str, meta: dict) -> dict:
    """Extract all fields from one cleaned document."""
    # --- metadata ---
    no_perkara = (P_NO_PERKARA.search(text) or [None, meta.get("no_perkara","")])[1] \
                  if P_NO_PERKARA.search(text) else meta.get("no_perkara","")

    tanggal_raw = P_TANGGAL.findall(text)
    tanggal     = parse_date(tanggal_raw[0]) if tanggal_raw else meta.get("tanggal","")

    terdakwa_m  = P_TERDAKWA.search(text)
    terdakwa    = terdakwa_m.group(1).strip() if terdakwa_m else meta.get("terdakwa","")

    pasal_m     = P_PASAL.search(text)
    pasal       = pasal_m.group(0).strip() if pasal_m else meta.get("pasal","")

    narkoba_m   = P_JENIS_NARK.search(text)
    jenis_nark  = narkoba_m.group(0).lower() if narkoba_m else meta.get("jenis_narkoba","")

    berat_m     = P_BERAT.search(text)
    berat       = berat_m.group(0) if berat_m else meta.get("berat","")

    amar_m      = P_AMAR.search(text)
    amar_raw    = amar_m.group(0).strip() if amar_m else meta.get("amar_putusan","")

    # --- key content ---
    ringkasan_dakwaan = extract_section(text, P_DAKWAAN)
    ringkasan_fakta   = extract_section(text, P_FAKTA)
    argumen_hukum     = extract_section(text, P_ANALISIS)

    # --- feature engineering ---
    text_lower  = text.lower()
    wc          = word_count(text)

    # Simple categorical label for ML
    label = meta.get("label_putusan", "pidana penjara")

    return {
        "case_id"           : case_id,
        "filename"          : fname,
        "no_perkara"        : no_perkara,
        "tanggal"           : tanggal,
        "pengadilan"        : meta.get("pengadilan",""),
        "jenis_perkara"     : "Pidana Khusus Narkotika & Psikotropika",
        "terdakwa"          : terdakwa,
        "pasal"             : pasal,
        "jenis_narkoba"     : jenis_nark,
        "berat_barang_bukti": berat,
        "amar_putusan"      : amar_raw,
        "label_putusan"     : label,
        "ringkasan_dakwaan" : ringkasan_dakwaan,
        "ringkasan_fakta"   : ringkasan_fakta,
        "argumen_hukum"     : argumen_hukum,
        "word_count"        : wc,
        "text_full"         : text,
    }


def build_representation():
    # Load metadata generated in stage 0
    with open(META_JSON, "r", encoding="utf-8") as f:
        meta_list = json.load(f)
    meta_by_id = {m["case_id"]: m for m in meta_list}

    files   = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".txt"))
    records = []

    for i, fname in enumerate(files):
        case_id = i + 1
        fpath   = os.path.join(RAW_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        meta   = meta_by_id.get(case_id, {})
        record = extract_record(case_id, fname, text, meta)
        records.append(record)
        print(f"  [+] {fname}  →  {record['terdakwa']} | {record['jenis_narkoba']} | {record['label_putusan']}")

    df = pd.DataFrame(records)

    # ── Save CSV ──
    csv_path = os.path.join(PROC_DIR, "cases.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ CSV tersimpan → {csv_path}  ({len(df)} baris)")

    # ── Save JSON (without text_full for readability) ──
    json_records = df.drop(columns=["text_full"]).to_dict(orient="records")
    json_path    = os.path.join(PROC_DIR, "cases.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON tersimpan → {json_path}")

    # ── Basic stats ──
    print("\n── Distribusi Label Putusan ──")
    print(df["label_putusan"].value_counts().to_string())
    print("\n── Distribusi Jenis Narkoba ──")
    print(df["jenis_narkoba"].value_counts().to_string())
    print(f"\n── Rata-rata Word Count: {df['word_count'].mean():.0f} kata ──")

    return df


if __name__ == "__main__":
    df = build_representation()
    print("\n=== Tahap 2 Selesai: Case Representation ===")
    print(df[["case_id","no_perkara","tanggal","terdakwa","pasal","label_putusan"]].to_string())
