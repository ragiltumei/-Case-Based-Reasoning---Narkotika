"""
generate_data.py
Generates 35 realistic synthetic Indonesian Supreme Court (MA RI) 
narcotics case decisions for CBR pipeline demo.
In a real project, replace this with actual scraped PDFs from 
https://putusan3.mahkamahagung.go.id/
"""
import os, json, random, re
from datetime import datetime, timedelta

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Vocabulary pools ──────────────────────────────────────────────────────────
PENGADILAN = [
    "Pengadilan Negeri Surabaya","Pengadilan Negeri Jakarta Selatan",
    "Pengadilan Negeri Medan","Pengadilan Negeri Makassar",
    "Pengadilan Negeri Bandung","Pengadilan Negeri Semarang",
    "Pengadilan Negeri Denpasar","Pengadilan Negeri Palembang",
]
NAMA_TERDAKWA = [
    "Ahmad Fauzi","Budi Santoso","Cipto Haryono","Dedi Kurniawan",
    "Eko Prasetyo","Fahri Maulana","Gunawan Hidayat","Hendra Wijaya",
    "Irwan Saputra","Joko Susilo","Kurnia Ramadhan","Lukman Hakim",
    "Muhamad Rizky","Nanang Setiawan","Oki Firmansyah","Pandu Wibowo",
    "Qori Abdillah","Randi Pratama","Sandi Nugroho","Taufiq Hidayat",
    "Umar Syarif","Vicky Andrianto","Wahyu Setiadi","Xander Kusuma",
    "Yusuf Maulana","Zaenal Arifin","Abdul Hakim","Bambang Riyanto",
    "Cahyo Purnomo","Dani Saputra","Erwin Susanto","Feri Kurniawan",
    "Gilang Saputra","Hadi Wijaya","Indra Maulana",
]
JAKSA = [
    "Dr. Bambang Harimurti, S.H., M.H.",
    "Siti Rahayu, S.H.",
    "Ahmad Ridwan, S.H., M.H.",
    "Dewi Kusuma, S.H.",
    "Hendra Basuki, S.H., M.H.",
]
HAKIM_KETUA = [
    "H. Surya Dharma, S.H., M.H.",
    "Dr. Retno Wulandari, S.H., M.Hum.",
    "Bambang Sudibyo, S.H.",
    "Sari Dewi, S.H., M.H.",
    "Agus Santoso, S.H., M.H.",
]
HAKIM_ANGGOTA = [
    ["Rina Susanti, S.H.", "Dedi Prasetyo, S.H., M.H."],
    ["Tono Wibowo, S.H.", "Sri Mulyani, S.H."],
    ["Hasan Basri, S.H., M.H.", "Nurhayati, S.H."],
    ["Wahid Suprapto, S.H.", "Endang Sulistyo, S.H., M.H."],
    ["Rudi Hartono, S.H.", "Fitri Handayani, S.H., M.H."],
]
JENIS_NARKOBA = [
    ("sabu-sabu (metamfetamina)","methamphetamine"),
    ("heroin","heroin"),
    ("ganja (cannabis)","cannabis"),
    ("ekstasi (MDMA)","MDMA"),
    ("kokain","cocaine"),
]
BERAT_SATUAN = [
    ("gram","g"), ("kilogram","kg"), ("paket kecil","paket"),
]

PASAL_OPTIONS = [
    ("114 ayat (1) UU No.35 Tahun 2009 tentang Narkotika",
     "terbukti menjadi perantara jual beli narkotika Golongan I"),
    ("114 ayat (2) UU No.35 Tahun 2009 tentang Narkotika",
     "terbukti menjadi perantara jual beli narkotika Golongan I dalam jumlah besar"),
    ("112 ayat (1) UU No.35 Tahun 2009 tentang Narkotika",
     "terbukti memiliki, menyimpan, menguasai narkotika Golongan I bukan tanaman"),
    ("112 ayat (2) UU No.35 Tahun 2009 tentang Narkotika",
     "terbukti memiliki narkotika Golongan I bukan tanaman melebihi 5 gram"),
    ("111 ayat (1) UU No.35 Tahun 2009 tentang Narkotika",
     "terbukti menanam, memelihara, memiliki narkotika Golongan I berupa tanaman"),
    ("132 ayat (1) UU No.35 Tahun 2009 tentang Narkotika",
     "terbukti melakukan permufakatan jahat untuk melakukan tindak pidana narkotika"),
]

PUTUSAN_JENIS = [
    ("penjara {tahun} tahun dan denda Rp {denda}.000.000 subsidair {sub} bulan kurungan",
     "pidana penjara"),
    ("penjara {tahun} tahun {bulan} bulan dan denda Rp {denda}.000.000 subsidair {sub} bulan",
     "pidana penjara"),
    ("penjara seumur hidup dan denda Rp {denda}.000.000 subsidair {sub} bulan kurungan",
     "penjara seumur hidup"),
    ("mati", "pidana mati"),
]

def random_date(start_year=2020, end_year=2024):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")

def make_no_perkara(i, tgl):
    yr = tgl[:4]
    return f"{100+i}/Pid.Sus/{yr}/PN.{random.choice(['Sby','Jkt.Sel','Mdn','Mks','Bdg'])}"

def build_putusan(i):
    tgl   = random_date()
    nama  = NAMA_TERDAKWA[i]
    narkoba, narkoba_en = random.choice(JENIS_NARKOBA)
    berat_val = round(random.uniform(0.5, 2500.0), 2)
    berat_sat = random.choice(BERAT_SATUAN)[0]
    pasal, dakwaan_singkat = random.choice(PASAL_OPTIONS)
    jaksa = random.choice(JAKSA)
    hakim_k = random.choice(HAKIM_KETUA)
    hakim_a = random.choice(HAKIM_ANGGOTA)
    pengadilan = random.choice(PENGADILAN)
    no_perkara = make_no_perkara(i+1, tgl)

    tahun_pid = random.randint(4, 20)
    bulan_pid = random.randint(0, 11)
    denda     = random.choice([800, 1000, 1500, 2000])
    sub       = random.randint(3, 6)

    putusan_template, label_putusan = random.choice(PUTUSAN_JENIS[:2])  # mostly penjara
    if berat_val > 1000:
        putusan_template, label_putusan = PUTUSAN_JENIS[2]  # seumur hidup
    amar = putusan_template.format(
        tahun=tahun_pid, bulan=bulan_pid, denda=denda, sub=sub
    )

    # Build full text
    dt = datetime.strptime(tgl, "%Y-%m-%d")
    tgl_fmt = f"{dt.day} {dt.strftime('%B %Y')}"
    text = f"""PUTUSAN
Nomor {no_perkara}

DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA

{pengadilan.upper()}

yang memeriksa dan mengadili perkara pidana khusus Narkotika dan Psikotropika pada 
tingkat pertama, telah menjatuhkan putusan sebagai berikut dalam perkara Terdakwa:

Nama Lengkap   : {nama}
Tempat Lahir   : {random.choice(['Surabaya','Jakarta','Medan','Makassar','Bandung'])}
Umur/Tanggal Lahir : {random.randint(18,55)} Tahun
Jenis Kelamin  : Laki-laki
Kebangsaan     : Indonesia
Agama          : {random.choice(['Islam','Kristen','Katolik'])}
Pekerjaan      : {random.choice(['Swasta','Wiraswasta','Buruh','Tidak Bekerja','Nelayan'])}
Alamat         : Jl. {random.choice(['Mawar','Melati','Kenanga','Anggrek'])} No.{random.randint(1,200)}, {random.choice(['Surabaya','Jakarta','Medan'])}

Terdakwa ditangkap oleh petugas Kepolisian pada tanggal {tgl_fmt} di {random.choice(['rumah','warung','parkiran','jalan raya'])} 
berdasarkan informasi dari masyarakat setempat.

DAKWAAN:

Bahwa Terdakwa {nama} pada hari {random.choice(['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu'])} 
tanggal {tgl_fmt}, bertempat di wilayah hukum {pengadilan}, 
{dakwaan_singkat} berupa {narkoba} seberat {berat_val} {berat_sat} 
yang telah disita sebagai barang bukti.

Perbuatan Terdakwa sebagaimana diatur dan diancam pidana dalam Pasal {pasal}.

FAKTA-FAKTA HUKUM:

Berdasarkan keterangan para saksi dan bukti-bukti yang diajukan di persidangan, 
diperoleh fakta-fakta hukum sebagai berikut:
1. Bahwa Terdakwa {nama} ditemukan sedang menguasai {narkoba} seberat {berat_val} {berat_sat}.
2. Bahwa barang bukti tersebut telah diuji di laboratorium dan terbukti mengandung 
   {narkoba_en} yang merupakan Narkotika Golongan I.
3. Bahwa Terdakwa tidak memiliki izin untuk memiliki, menyimpan, menguasai narkotika tersebut.
4. Bahwa Terdakwa sebelumnya pernah/tidak pernah dihukum karena melakukan tindak pidana.

Keterangan Saksi:
- Saksi Briptu {random.choice(['Handoko','Suryo','Rudi','Andi'])} (anggota kepolisian yang melakukan penangkapan) 
  menerangkan bahwa pada saat penangkapan ditemukan {narkoba} dalam penguasaan Terdakwa.
- Saksi {random.choice(['Suwandi','Parman','Tobing','Effendi'])} menerangkan bahwa ia mengenal Terdakwa sebagai 
  warga setempat dan tidak mengetahui kegiatan Terdakwa yang melanggar hukum.

Keterangan Terdakwa:
Terdakwa pada pokoknya menerangkan bahwa ia memang memiliki {narkoba} tersebut namun 
mengaku hanya sebagai pemakai/pengguna bukan pengedar.

ANALISIS HUKUM:

Majelis Hakim berpendapat bahwa unsur-unsur Pasal {pasal} telah terpenuhi berdasarkan:
1. Unsur "tanpa hak": Terdakwa tidak memiliki izin dari pejabat berwenang.
2. Unsur "memiliki/menguasai": Terbukti dari keterangan saksi dan temuan barang bukti.
3. Unsur narkotika Golongan I: Terbukti dari hasil uji laboratorium.

Hal-hal yang memberatkan:
- Perbuatan Terdakwa bertentangan dengan program pemberantasan narkoba.
- Narkoba merusak generasi bangsa.

Hal-hal yang meringankan:
- Terdakwa bersikap sopan di persidangan.
- Terdakwa mengakui perbuatannya.
- {random.choice(['Terdakwa belum pernah dihukum.','Terdakwa menyesali perbuatannya.'])}

MENGADILI:

1. Menyatakan Terdakwa {nama} terbukti secara sah dan meyakinkan bersalah melakukan 
   tindak pidana Narkotika sebagaimana diatur dalam Pasal {pasal}.
2. Menjatuhkan pidana kepada Terdakwa dengan pidana {amar}.
3. Menetapkan masa penangkapan dan penahanan yang telah dijalani Terdakwa dikurangkan 
   seluruhnya dari pidana yang dijatuhkan.
4. Memerintahkan agar barang bukti berupa {narkoba} seberat {berat_val} {berat_sat} dirampas 
   untuk dimusnahkan.
5. Membebankan biaya perkara kepada Terdakwa sebesar Rp5.000,00 (lima ribu rupiah).

Demikian diputuskan dalam rapat permusyawaratan Majelis Hakim {pengadilan} 
pada hari {random.choice(['Senin','Selasa','Rabu','Kamis','Jumat'])} tanggal {tgl_fmt}.

Hakim Ketua  : {hakim_k}
Hakim Anggota: {hakim_a[0]}
             : {hakim_a[1]}
Panitera     : {random.choice(['Slamet Riyadi, S.H.','Dewi Anggraini, S.H.','Budi Hartono, S.H.'])}

Jaksa Penuntut Umum: {jaksa}
"""
    meta = {
        "case_id"       : i + 1,
        "no_perkara"    : no_perkara,
        "tanggal"       : tgl,
        "pengadilan"    : pengadilan,
        "jenis_perkara" : "Pidana Khusus Narkotika & Psikotropika",
        "terdakwa"      : nama,
        "pasal"         : pasal,
        "jenis_narkoba" : narkoba,
        "berat"         : f"{berat_val} {berat_sat}",
        "amar_putusan"  : amar,
        "label_putusan" : label_putusan,
        "jaksa"         : jaksa,
        "hakim_ketua"   : hakim_k,
    }
    return text, meta

def main():
    all_meta = []
    for i in range(35):
        text, meta = build_putusan(i)
        fname = f"case_{i+1:03d}.txt"
        fpath = os.path.join(RAW_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text)
        all_meta.append(meta)
        print(f"  [+] {fname}  |  {meta['no_perkara']}  |  {meta['terdakwa']}")

    meta_path = os.path.join(PROCESSED_DIR, "cases_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    print(f"\n✓ {len(all_meta)} dokumen putusan dihasilkan → data/raw/")
    print(f"✓ Metadata tersimpan → {meta_path}")

if __name__ == "__main__":
    main()
