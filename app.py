# ============================================================
# TIMESTAMP REMOVER - STREAMLIT UI (BATCH + ZIP) - SMOOTH v3
# ============================================================
#
# Kebalikan dari "Timestamp Generator": app ini MENGHAPUS
# watermark timestamp gaya GPS Map Camera (koordinat, lokasi,
# tanggal & jam, teks putih dengan outline hitam di pojok kiri
# bawah) dari foto.
#
# ============================================================
# APA YANG BARU DI v3: DUA METODE
# ============================================================
#
# 1) METODE KLASIK (offline, tanpa API key)
#    - Deteksi: heuristik piksel putih terang + dekat outline
#      tajam di pita bawah foto (lihat detect_timestamp_mask).
#    - Rekonstruksi: cv2.inpaint (TELEA/NS), opsional multi-scale
#      + feathered blending ("Mode Halus") supaya lebih smooth.
#    - Ini algoritma klasik, BUKAN AI generatif. Untuk pola
#      background yang rumit, hasilnya punya batas.
#
# 2) METODE AI - GEMINI ("Nano Banana", gemini-2.5-flash-image)
#    - Satu API call: kirim foto + instruksi teks, model
#      langsung mengembalikan foto hasil edit (deteksi lokasi
#      overlay & rekonstruksi background ditangani model itu
#      sendiri, reasoning-based, tanpa mask manual).
#    - Butuh API key Gemini (Google AI Studio / Vertex) dan
#      akses internet dari environment yang menjalankan app ini.
#    - Install dulu: pip install google-genai
#
# Kedua metode bisa dipilih di sidebar. Untuk metode AI, mask
# preview tidak tersedia (tidak ada mask eksplisit -- modelnya
# langsung menghasilkan gambar akhir).
#
# BATCH:
#   - Bisa upload banyak foto sekaligus, ATAU upload 1 file .zip
#     berisi banyak foto.
#   - Semua foto diproses dengan metode & parameter yang sama,
#     lalu bisa didownload satu-satu atau sekaligus dalam .zip.
# ============================================================


# ============================================================
# IMPORT
# ============================================================

import io
import os
import time
import zipfile

import cv2
import numpy as np
import streamlit as st

from PIL import Image, ImageOps


# ============================================================
# ============================================================
# KONFIGURASI DEFAULT
# ============================================================
# ============================================================

DEFAULT_TOP_PCT = 0.72       # mulai cari dari 72% tinggi foto ke bawah
DEFAULT_LEFT_PCT = 0.0
DEFAULT_RIGHT_PCT = 1.0
DEFAULT_BOTTOM_PCT = 1.0

DEFAULT_WHITE_THRESH = 190   # ambang piksel dianggap "putih terang"
DEFAULT_EDGE_THRESH = 25     # ambang kekuatan tepi (outline)
DEFAULT_DILATE_PX = 7        # pelebaran mask akhir (px) sebelum inpaint
DEFAULT_INPAINT_RADIUS = 5

MIN_BLOB_AREA = 15            # buang noda mask yang terlalu kecil

DEFAULT_SMOOTH_MODE = True
DEFAULT_SMOOTH_SCALES = (0.25, 0.5, 1.0)   # dari kasar -> halus
DEFAULT_FEATHER_PX = 3        # kelembutan tepi (gaussian sigma, px)

# --- Gemini / Nano Banana ---
GEMINI_MODEL_NAME = "gemini-flash-lite-latest"

DEFAULT_GEMINI_PROMPT = (
    "This photo has a GPS-Map-Camera style timestamp overlay in the "
    "bottom-left corner: white text with a black outline showing GPS "
    "coordinates, a location name/address, and a date & time, usually "
    "sitting on a small semi-transparent dark strip. Remove that entire "
    "overlay completely and reconstruct the background behind it so it "
    "looks natural and seamless, matching the surrounding textures, "
    "lighting, and colors. Do not alter, crop, or recompose any other "
    "part of the image -- keep everything else pixel-for-pixel identical "
    "in composition. Output only the edited photo."
)

GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_BACKOFF_SEC = 2.0


# ============================================================
# ============================================================
# LOAD FOTO DENGAN FIX ORIENTASI EXIF
# ============================================================
# ============================================================

def load_pil_image_fixed_orientation(file_bytes):

    pil_image = Image.open(io.BytesIO(file_bytes))

    pil_image = ImageOps.exif_transpose(pil_image)

    pil_image = pil_image.convert("RGB")

    return pil_image


# ============================================================
# ============================================================
# METODE KLASIK: DETEKSI TIMESTAMP -> MASK
# ============================================================
# ============================================================

def detect_timestamp_mask(
    image_bgr,
    top_pct=DEFAULT_TOP_PCT,
    left_pct=DEFAULT_LEFT_PCT,
    right_pct=DEFAULT_RIGHT_PCT,
    bottom_pct=DEFAULT_BOTTOM_PCT,
    white_thresh=DEFAULT_WHITE_THRESH,
    edge_thresh=DEFAULT_EDGE_THRESH,
    dilate_px=DEFAULT_DILATE_PX,
):

    height, width = image_bgr.shape[:2]

    top = int(height * top_pct)
    bottom = int(height * bottom_pct)
    left = int(width * left_pct)
    right = int(width * right_pct)

    roi = image_bgr[top:bottom, left:right]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # Kekuatan tepi lokal (outline teks = kontras sangat tajam)
    # --------------------------------------------------------

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    edge_magnitude = np.abs(laplacian)

    edge_magnitude = cv2.normalize(
        edge_magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, edge_mask = cv2.threshold(
        edge_magnitude, edge_thresh, 255, cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # Piksel putih terang (isi teks)
    # --------------------------------------------------------

    _, white_mask = cv2.threshold(
        gray, white_thresh, 255, cv2.THRESH_BINARY
    )

    edge_dilated = cv2.dilate(
        edge_mask, np.ones((5, 5), np.uint8), iterations=1
    )

    # Teks = putih terang YANG berdekatan dengan tepi tajam
    # (menyaring area putih polos besar seperti lantai/tembok)
    text_mask = cv2.bitwise_and(white_mask, edge_dilated)

    # Tangkap juga outline hitamnya (tepi tajam yang menempel
    # ke area teks putih di atas)
    text_dilated = cv2.dilate(
        text_mask, np.ones((9, 9), np.uint8), iterations=1
    )

    outline_mask = cv2.bitwise_and(edge_mask, text_dilated)

    combined_mask = cv2.bitwise_or(text_mask, outline_mask)

    # Sambung celah antar-huruf/antar-baris, lalu beri margin aman
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
    )

    combined_mask = cv2.dilate(
        combined_mask,
        np.ones((dilate_px, dilate_px), np.uint8),
        iterations=1,
    )

    # Buang noda kecil (bukan bagian teks)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined_mask, connectivity=8
    )

    clean_mask = np.zeros_like(combined_mask)

    for label_index in range(1, num_labels):

        area = stats[label_index, cv2.CC_STAT_AREA]

        if area >= MIN_BLOB_AREA:

            clean_mask[labels == label_index] = 255

    full_mask = np.zeros((height, width), dtype=np.uint8)

    full_mask[top:bottom, left:right] = clean_mask

    return full_mask


# ============================================================
# ============================================================
# METODE KLASIK - REKONSTRUKSI: MODE CEPAT (1 pass)
# ============================================================
# ============================================================

def reconstruct_fast(image_bgr, mask, inpaint_radius=DEFAULT_INPAINT_RADIUS):

    return cv2.inpaint(image_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)


# ============================================================
# ============================================================
# METODE KLASIK - REKONSTRUKSI: MODE HALUS
# (multi-scale + feathered blending)
# ============================================================
# ============================================================

def reconstruct_smooth(
    image_bgr,
    mask,
    inpaint_radius=DEFAULT_INPAINT_RADIUS,
    scales=DEFAULT_SMOOTH_SCALES,
    feather_px=DEFAULT_FEATHER_PX,
):

    height, width = image_bgr.shape[:2]

    result = image_bgr.copy()

    for scale in scales:

        small_w = max(1, int(width * scale))
        small_h = max(1, int(height * scale))

        img_small = cv2.resize(
            result, (small_w, small_h), interpolation=cv2.INTER_AREA
        )

        mask_small = cv2.resize(
            mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST
        )

        _, mask_small = cv2.threshold(
            mask_small, 127, 255, cv2.THRESH_BINARY
        )

        if not np.any(mask_small):
            continue

        # skala kasar -> INPAINT_NS (lebih smooth untuk area luas)
        # skala penuh -> INPAINT_TELEA (lebih tajam untuk detail)
        method = cv2.INPAINT_NS if scale < 1.0 else cv2.INPAINT_TELEA

        inpainted_small = cv2.inpaint(
            img_small, mask_small, inpaint_radius, method
        )

        inpainted_up = cv2.resize(
            inpainted_small, (width, height), interpolation=cv2.INTER_CUBIC
        )

        # alpha lembut = mask asli (resolusi penuh) di-blur Gaussian
        alpha = mask.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=feather_px)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]

        result = (
            inpainted_up.astype(np.float32) * alpha
            + result.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)

    return result


# ============================================================
# ============================================================
# METODE KLASIK: FUNGSI UTAMA (DETEKSI + HAPUS)
# ============================================================
# ============================================================

def remove_timestamp_classic(
    image_bgr,
    top_pct=DEFAULT_TOP_PCT,
    left_pct=DEFAULT_LEFT_PCT,
    right_pct=DEFAULT_RIGHT_PCT,
    bottom_pct=DEFAULT_BOTTOM_PCT,
    white_thresh=DEFAULT_WHITE_THRESH,
    edge_thresh=DEFAULT_EDGE_THRESH,
    dilate_px=DEFAULT_DILATE_PX,
    inpaint_radius=DEFAULT_INPAINT_RADIUS,
    smooth_mode=DEFAULT_SMOOTH_MODE,
    feather_px=DEFAULT_FEATHER_PX,
):

    full_mask = detect_timestamp_mask(
        image_bgr,
        top_pct=top_pct,
        left_pct=left_pct,
        right_pct=right_pct,
        bottom_pct=bottom_pct,
        white_thresh=white_thresh,
        edge_thresh=edge_thresh,
        dilate_px=dilate_px,
    )

    detected_px = int(np.count_nonzero(full_mask))

    if detected_px == 0:

        return image_bgr.copy(), full_mask, detected_px

    if smooth_mode:

        reconstructed = reconstruct_smooth(
            image_bgr,
            full_mask,
            inpaint_radius=inpaint_radius,
            feather_px=feather_px,
        )

    else:

        reconstructed = reconstruct_fast(
            image_bgr, full_mask, inpaint_radius=inpaint_radius
        )

    return reconstructed, full_mask, detected_px


# ============================================================
# ============================================================
# METODE AI: GEMINI ("NANO BANANA") DETEKSI + INPAINTING
# ============================================================
# ============================================================
#
# Perlu paket: pip install google-genai
#
# Satu panggilan generate_content dengan input gambar + prompt
# teks. Modelnya (gemini-2.5-flash-image) mengembalikan bagian
# gambar (inline_data) berisi foto hasil edit -- tidak ada mask
# eksplisit yang kita kontrol, semua ditangani model.
# ============================================================

def get_gemini_client(api_key):

    from google import genai

    return genai.Client(api_key=api_key)


def remove_timestamp_gemini(
    pil_image,
    api_key,
    prompt=DEFAULT_GEMINI_PROMPT,
    max_retries=GEMINI_MAX_RETRIES,
):
    """
    Kirim 1 foto ke Gemini image model untuk menghapus overlay
    timestamp. Mengembalikan (result_pil_image_or_None, error_str_or_None).
    """

    try:
        from google.genai import types  # noqa: F401
    except ImportError as import_error:

        return None, (
            "Paket 'google-genai' belum terinstall. Jalankan: "
            "pip install google-genai  --  "
            f"({import_error})"
        )

    last_error = None

    for attempt in range(1, max_retries + 1):

        try:

            client = get_gemini_client(api_key)

            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[pil_image, prompt],
            )

            candidates = getattr(response, "candidates", None)

            if not candidates:
                last_error = "Response Gemini kosong (tidak ada candidates)."
                continue

            parts = candidates[0].content.parts

            for part in parts:

                inline_data = getattr(part, "inline_data", None)

                if inline_data is not None and inline_data.data:

                    result_image = Image.open(
                        io.BytesIO(inline_data.data)
                    ).convert("RGB")

                    return result_image, None

            # Kalau sampai sini, tidak ada bagian gambar di response.
            # Mungkin model menolak / hanya membalas teks.
            text_parts = [
                getattr(p, "text", None) for p in parts
            ]
            text_joined = " ".join(t for t in text_parts if t)

            last_error = (
                "Gemini tidak mengembalikan gambar hasil edit."
                + (f" Pesan model: {text_joined}" if text_joined else "")
            )

        except Exception as call_error:  # noqa: BLE001

            last_error = str(call_error)

            if attempt < max_retries:

                time.sleep(GEMINI_RETRY_BACKOFF_SEC * attempt)

    return None, last_error


# ============================================================
# ============================================================
# HELPER: ENCODE HASIL KE BYTES
# ============================================================
# ============================================================

def encode_image_bytes_from_bgr(image_bgr, ext):

    if ext.lower() in [".jpg", ".jpeg"]:

        ok, buf = cv2.imencode(
            ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

    else:

        ok, buf = cv2.imencode(".png", image_bgr)

    if not ok:

        return None

    return buf.tobytes()


def encode_image_bytes_from_pil(pil_image, ext):

    buf = io.BytesIO()

    if ext.lower() in [".jpg", ".jpeg"]:

        pil_image.save(buf, format="JPEG", quality=95)

    else:

        pil_image.save(buf, format="PNG")

    return buf.getvalue()


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def collect_input_files(uploaded_files):
    """
    Terima list file dari st.file_uploader (foto biasa dan/atau
    .zip). Kembalikan list of (filename, raw_bytes) untuk semua
    foto yang ditemukan.
    """

    collected = []

    for uploaded in uploaded_files:

        name_lower = uploaded.name.lower()

        if name_lower.endswith(".zip"):

            with zipfile.ZipFile(uploaded) as zip_file:

                for zip_info in zip_file.infolist():

                    if zip_info.is_dir():
                        continue

                    inner_name = zip_info.filename

                    if not inner_name.lower().endswith(IMAGE_EXTS):
                        continue

                    if "__MACOSX" in inner_name:
                        continue

                    with zip_file.open(zip_info) as inner_file:

                        collected.append(
                            (
                                os.path.basename(inner_name),
                                inner_file.read(),
                            )
                        )

        elif name_lower.endswith(IMAGE_EXTS):

            collected.append((uploaded.name, uploaded.getvalue()))

    return collected


# ============================================================
# ============================================================
# STREAMLIT UI
# ============================================================
# ============================================================

st.set_page_config(
    page_title="Timestamp Remover | GPS Camera Style",
    page_icon="🧹",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("## 🧹 Timestamp Remover")
st.caption(
    "Hapus otomatis watermark timestamp GPS Map Camera "
    "(koordinat, lokasi, tanggal & jam) dari foto. "
    "Bisa upload banyak foto sekaligus atau 1 file .zip."
)

# ------------------------------------------------------------
# SIDEBAR: PILIH METODE
# ------------------------------------------------------------

with st.sidebar:

    st.markdown("### 🧠 Metode")

    method = st.radio(
        "Pilih metode deteksi & rekonstruksi",
        options=["Klasik (offline, cv2.inpaint)", "AI - Gemini (Nano Banana)"],
        index=0,
        help="Klasik: heuristik piksel + cv2.inpaint, jalan offline. "
        "AI: kirim foto ke Gemini image model, deteksi & rekonstruksi "
        "ditangani model dalam satu API call. Butuh API key & internet.",
    )

    use_ai = method.startswith("AI")

    gemini_api_key = None
    gemini_prompt = DEFAULT_GEMINI_PROMPT

    if use_ai:

        st.markdown("### 🔑 Gemini API")

        gemini_api_key = st.text_input(
            "Gemini API key",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            help="Bisa juga di-set lewat environment variable GEMINI_API_KEY "
            "supaya tidak perlu diketik ulang.",
        )

        st.caption(
            "Butuh paket `google-genai` (`pip install google-genai`) dan "
            f"akses ke model `{GEMINI_MODEL_NAME}`. Setiap foto = 1 API "
            "call terpisah, jadi ada biaya & rate limit sesuai akun Gemini "
            "kamu."
        )

        with st.expander("✏️ Edit instruksi (prompt) ke Gemini"):

            gemini_prompt = st.text_area(
                "Prompt",
                value=DEFAULT_GEMINI_PROMPT,
                height=160,
            )

    st.markdown("### ⚙️ Parameter Deteksi (Klasik)")

    st.caption(
        "Hanya dipakai kalau metode = Klasik. Default biasanya sudah "
        "cukup; atur ulang kalau hasil kurang bersih."
    )

    top_pct = st.slider(
        "Mulai area deteksi dari (% tinggi foto)",
        min_value=40,
        max_value=95,
        value=int(DEFAULT_TOP_PCT * 100),
        step=1,
        disabled=use_ai,
        help="Timestamp GPS Map Camera selalu di pojok kiri-bawah. "
        "Naikkan angka ini kalau timestamp-nya cuma 1-2 baris pendek "
        "(area deteksi jadi lebih sempit / lebih ke bawah).",
    ) / 100.0

    white_thresh = st.slider(
        "Ambang kecerahan teks (white threshold)",
        min_value=140,
        max_value=250,
        value=DEFAULT_WHITE_THRESH,
        step=5,
        disabled=use_ai,
    )

    edge_thresh = st.slider(
        "Sensitivitas outline (edge threshold)",
        min_value=5,
        max_value=80,
        value=DEFAULT_EDGE_THRESH,
        step=5,
        disabled=use_ai,
        help="Turunkan kalau outline teks tipis/tidak terdeteksi. "
        "Naikkan kalau area lain (misal tekstur lantai) ikut terdeteksi.",
    )

    dilate_px = st.slider(
        "Margin aman sekitar teks (px)",
        min_value=1,
        max_value=20,
        value=DEFAULT_DILATE_PX,
        step=1,
        disabled=use_ai,
    )

    inpaint_radius = st.slider(
        "Radius rekonstruksi latar (inpaint radius)",
        min_value=1,
        max_value=15,
        value=DEFAULT_INPAINT_RADIUS,
        step=1,
        disabled=use_ai,
    )

    st.markdown("### ✨ Kehalusan Hasil (Klasik)")

    smooth_mode = st.checkbox(
        "Mode Halus (multi-scale + feathered blending)",
        value=DEFAULT_SMOOTH_MODE,
        disabled=use_ai,
        help="Rekonstruksi dilakukan berlapis dari resolusi kecil ke "
        "besar, lalu digabung dengan transisi lembut di tepi. Matikan "
        "untuk kembali ke mode cepat (1 pass). Tidak berlaku untuk "
        "metode AI.",
    )

    feather_px = DEFAULT_FEATHER_PX

    if smooth_mode and not use_ai:

        feather_px = st.slider(
            "Kelembutan tepi (feather, px)",
            min_value=0,
            max_value=10,
            value=DEFAULT_FEATHER_PX,
            step=1,
            help="Makin besar, transisi antara area rekonstruksi dan "
            "area asli makin lembut (mengurangi garis 'jahitan').",
        )

    show_mask_preview = st.checkbox(
        "Tampilkan preview area yang terdeteksi (mask)",
        value=True,
        disabled=use_ai,
        help="Hanya tersedia untuk metode Klasik -- metode AI tidak "
        "menghasilkan mask eksplisit.",
    )


# ------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------

st.markdown("### 1️⃣ Upload Foto")

uploaded_files = st.file_uploader(
    "Pilih foto (JPG/PNG) atau upload 1 file .zip berisi banyak foto",
    type=["jpg", "jpeg", "png", "zip"],
    accept_multiple_files=True,
    help="Bisa pilih banyak file foto sekaligus, atau cukup 1 file .zip.",
)

if not uploaded_files:

    st.info("📷 Belum ada foto/zip yang diupload.")

    st.stop()

input_files = collect_input_files(uploaded_files)

if not input_files:

    st.warning(
        "⚠️ Tidak ada foto (JPG/PNG) yang ditemukan dari file yang diupload."
    )

    st.stop()

st.success(f"✅ {len(input_files)} foto siap diproses.")

if use_ai and not gemini_api_key:

    st.warning(
        "⚠️ Metode AI dipilih tapi API key Gemini belum diisi (lihat sidebar)."
    )


# ------------------------------------------------------------
# PROSES
# ------------------------------------------------------------

st.markdown("### 2️⃣ Proses & Preview")

process_clicked = st.button(
    "🚀 HAPUS TIMESTAMP DARI SEMUA FOTO",
    type="primary",
    use_container_width=True,
    disabled=(use_ai and not gemini_api_key),
)

if process_clicked:

    results = []

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    for index, (filename, raw_bytes) in enumerate(input_files):

        status_text.caption(f"Memproses `{filename}` ({index + 1}/{len(input_files)})...")

        try:

            pil_image = load_pil_image_fixed_orientation(raw_bytes)

            name, ext = os.path.splitext(filename)

            if not ext:
                ext = ".jpg"

            output_name = name + "_clean" + ext

            if use_ai:

                # ------------------------------------------------
                # METODE AI (Gemini)
                # ------------------------------------------------

                result_pil, error = remove_timestamp_gemini(
                    pil_image,
                    api_key=gemini_api_key,
                    prompt=gemini_prompt,
                )

                if error is not None:

                    st.error(f"❌ Gagal memproses `{filename}` lewat Gemini: {error}")
                    continue

                original_rgb = np.array(pil_image)
                result_rgb = np.array(result_pil)

                encoded_bytes = encode_image_bytes_from_pil(result_pil, ext)

                results.append(
                    {
                        "filename": filename,
                        "output_name": output_name,
                        "original_rgb": original_rgb,
                        "result_rgb": result_rgb,
                        "mask": None,
                        "detected_px": None,
                        "encoded_bytes": encoded_bytes,
                        "mime": "image/jpeg" if ext.lower() in [".jpg", ".jpeg"] else "image/png",
                        "method": "ai",
                    }
                )

            else:

                # ------------------------------------------------
                # METODE KLASIK (cv2.inpaint)
                # ------------------------------------------------

                image_rgb = np.array(pil_image)

                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                result_bgr, mask, detected_px = remove_timestamp_classic(
                    image_bgr,
                    top_pct=top_pct,
                    white_thresh=white_thresh,
                    edge_thresh=edge_thresh,
                    dilate_px=dilate_px,
                    inpaint_radius=inpaint_radius,
                    smooth_mode=smooth_mode,
                    feather_px=feather_px,
                )

                encoded_bytes = encode_image_bytes_from_bgr(result_bgr, ext)

                results.append(
                    {
                        "filename": filename,
                        "output_name": output_name,
                        "original_rgb": image_rgb,
                        "result_rgb": cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB),
                        "mask": mask,
                        "detected_px": detected_px,
                        "encoded_bytes": encoded_bytes,
                        "mime": "image/jpeg" if ext.lower() in [".jpg", ".jpeg"] else "image/png",
                        "method": "classic",
                    }
                )

        except Exception as error:

            st.error(f"❌ Gagal memproses `{filename}`: {error}")

        progress_bar.progress((index + 1) / len(input_files))

    status_text.empty()

    st.session_state["remover_results"] = results


# ------------------------------------------------------------
# HASIL
# ------------------------------------------------------------

results = st.session_state.get("remover_results")

if results:

    st.markdown("---")
    st.markdown("### ✨ Hasil")

    classic_results = [r for r in results if r["method"] == "classic"]

    zero_detect_count = sum(
        1 for r in classic_results if r["detected_px"] == 0
    )

    if zero_detect_count > 0:

        st.warning(
            f"⚠️ {zero_detect_count} dari {len(classic_results)} foto (metode "
            "Klasik) TIDAK terdeteksi ada timestamp di area yang dicari. "
            "Coba longgarkan parameter di sidebar (misal turunkan '% "
            "tinggi foto' atau 'edge threshold'), atau coba metode AI."
        )

    for r in results:

        with st.container(border=True):

            st.markdown(f"**{r['filename']}**  \n`metode: {r['method']}`")

            col_before, col_after = st.columns(2)

            with col_before:

                st.image(
                    r["original_rgb"],
                    caption="Sebelum",
                    use_container_width=True,
                )

            with col_after:

                st.image(
                    r["result_rgb"],
                    caption="Sesudah",
                    use_container_width=True,
                )

            if r["method"] == "classic" and show_mask_preview and r["mask"] is not None:

                with st.expander("🔍 Area yang terdeteksi & dihapus (mask)"):

                    st.image(
                        r["mask"],
                        caption=f"Terdeteksi ~{r['detected_px']:,} piksel",
                        use_container_width=True,
                    )

            st.download_button(
                label=f"⬇️ Download {r['output_name']}",
                data=r["encoded_bytes"],
                file_name=r["output_name"],
                mime=r["mime"],
                key=f"dl_{r['filename']}",
            )

    # ----------------------------------------------------
    # DOWNLOAD SEMUA SEKALIGUS (ZIP)
    # ----------------------------------------------------

    st.markdown("---")

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_out:

        for r in results:

            zip_out.writestr(r["output_name"], r["encoded_bytes"])

    st.download_button(
        label=f"⬇️ DOWNLOAD SEMUA ({len(results)} foto) SEBAGAI .ZIP",
        data=zip_buffer.getvalue(),
        file_name="foto_tanpa_timestamp.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "Metode Klasik: deteksi berbasis pola piksel (putih terang + outline "
    "tajam) di pita bawah foto, rekonstruksi pakai cv2.inpaint (opsional "
    "multi-scale + feathered blending) -- tanpa AI/API key. "
    f"Metode AI: satu panggilan ke `{GEMINI_MODEL_NAME}` (Gemini / "
    "Nano Banana) untuk deteksi & rekonstruksi sekaligus -- butuh API "
    "key & internet. Kalau hasil kurang pas, atur parameter/prompt di "
    "sidebar lalu proses ulang."
)
