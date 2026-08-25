# Timestamp Remover V2.9.1 — Streamlit

Versi Streamlit dari code Google Colab V2.9, dengan perbaikan pada restorasi garis lurus.

## Perbaikan utama

1. `cv2.HoughLinesP()` dinormalisasi dengan `reshape(-1, 4)`, sehingga kompatibel dengan output `(N,1,4)` maupun `(N,4)`.
2. `cv2.clipLine()` memakai rectangle `(x, y, width, height)`.
3. Restorasi garis dibungkus fail-safe. Jika restorasi gagal, hasil SimpleLama tetap dianggap sukses.
4. EasyOCR dan SimpleLama di-cache dengan `st.cache_resource`, sehingga tidak di-load ulang setiap rerun.
5. Multi-upload, preview before/after, download per foto, dan ZIP semua hasil.

## Jalankan lokal

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Upload seluruh folder ke GitHub, lalu pilih `app.py` sebagai entry point.

Catatan: CPU tetap bisa digunakan, tetapi SimpleLama lebih lambat dibanding GPU.
