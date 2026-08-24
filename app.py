# ============================================================
# TIMESTAMP REMOVER - STREAMLIT UI (BATCH + ZIP) - SMOOTH v2
# ============================================================
#
# Kebalikan dari "Timestamp Generator": app ini MENGHAPUS
# watermark timestamp gaya GPS Map Camera (koordinat, lokasi,
# tanggal & jam, teks putih dengan outline hitam di pojok kiri
# bawah) dari foto.
#
# CARA KERJA DETEKSI (tanpa AI/API key, murni image processing):
#   1. Hanya melihat area PITA BAWAH foto (persentase tinggi
#      bisa diatur), karena timestamp GPS Map Camera selalu di
#      pojok kiri bawah.
#   2. Di area itu, cari piksel PUTIH TERANG (isi teks) yang
#      posisinya berdekatan dengan tepi kontras tinggi (outline
#      hitam teks). Kombinasi "putih + nempel ke outline tajam"
#      inilah yang membedakan teks dari area putih polos biasa
#      (lantai keramik putih, tembok putih, baju putih, dll)
#      yang TIDAK dekat dengan outline tajam serapat itu.
#   3. Mask hasil deteksi di-dilate & direkonstruksi.
#
# ============================================================
# APA YANG BERUBAH DI v2 (SUPAYA LEBIH SMOOTH)
# ============================================================
#
# Versi lama memanggil cv2.inpaint() SATU KALI, langsung di
# resolusi penuh. cv2.inpaint (TELEA/NS) itu algoritma KLASIK,
# bukan AI generatif -- dia cuma "menjalarkan" warna dari tepi
# mask ke dalam berdasarkan tetangga terdekat. Untuk background
# bertekstur berulang (garis keramik, garis casing TV, paving)
# hasilnya sering blur / bergaris tidak nyambung, dan di tepi
# mask suka kelihatan "jahitan" (seam) karena transisinya tajam.
#
# Perbaikan yang diterapkan di v2 (tetap tanpa AI/API key,
# karena sandbox ini tidak ada akses internet untuk download
# model AI inpainting seperti LaMa / Stable Diffusion):
#
#   A. MULTI-SCALE INPAINTING
#      Inpaint dulu di resolusi kecil (25%, 50%) -> dapat pola
#      besar yang lebih koheren (garis keramik jadi lebih nyambung
#      karena "dilihat" dari jauh, bukan piksel-per-piksel).
#      Hasil itu di-upscale, lalu di resolusi penuh dilakukan
#      inpaint sekali lagi untuk menambah detail halus.
#
#   B. FEATHERED BLENDING (alpha blending lembut di tepi mask)
#      Setiap hasil inpaint dari tiap skala digabung ke gambar
#      pakai mask yang di-Gaussian-blur (jadi alpha 0..1 lembut),
#      bukan mask keras 0/1. Ini menghilangkan "garis potong"
#      di tepi area yang direkonstruksi.
#
#   C. KOMBINASI METODE
#      Pass resolusi kecil pakai INPAINT_NS (lebih smooth, bagus
#      untuk area luas/gradasi warna). Pass resolusi penuh pakai
#      INPAINT_TELEA (lebih tajam, bagus untuk detail/tekstur).
#
# Mode ini bisa dimatikan di sidebar ("Mode Halus") kalau mau
# balik ke inpaint satu-pass yang lebih cepat.
#
# CATATAN JUJUR:
#   Ini TETAP bukan AI generatif yang "paham" isi gambar seperti
#   LaMa/Stable-Diffusion inpainting. Untuk pola sangat kompleks
#   (misal motif lantai unik, wajah, objek 3D di area yang
#   ketiban teks), hasil klasik seperti ini punya batas. Kalau
#   butuh kualitas setara AI generatif, jalankan model open-source
#   seperti LaMa (https://github.com/advimman/lama) atau pipeline
#   Stable Diffusion inpainting di environment yang ada GPU +
#   akses internet untuk download weights-nya.
#
# BATCH:
#   - Bisa upload banyak foto sekaligus, ATAU upload 1 file .zip
#     berisi banyak foto.
#   - Semua foto diproses dengan parameter yang sama (bisa
#     diatur di sidebar), lalu bisa didownload satu-satu atau
#     sekaligus dalam .zip.
# ============================================================


# ============================================================
# IMPORT
# ============================================================

import io
import os
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
# DETEKSI TIMESTAMP -> MASK
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
# REKONSTRUKSI - MODE CEPAT (1 pass, seperti versi lama)
# ============================================================
# ============================================================

def reconstruct_fast(image_bgr, mask, inpaint_radius=DEFAULT_INPAINT_RADIUS):

    return cv2.inpaint(image_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)


# ============================================================
# ============================================================
# REKONSTRUKSI - MODE HALUS (multi-scale + feathered blending)
# ============================================================
# ============================================================
#
# Ide: dari resolusi paling kasar ke paling halus, tiap tahap
# melakukan inpaint lalu di-blend balik ke gambar hasil tahap
# sebelumnya pakai alpha yang lembut (mask di-Gaussian-blur).
# Tahap kasar menyumbang struktur/pola besar yang koheren,
# tahap halus menyumbang detail, dan feathering menghilangkan
# garis potong tajam di tepi area yang direkonstruksi.
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
# FUNGSI UTAMA: DETEKSI + HAPUS TIMESTAMP
# ============================================================
# ============================================================

def remove_timestamp(
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
# HELPER: ENCODE HASIL KE BYTES
# ============================================================
# ============================================================

def encode_image_bytes(image_bgr, ext):

    if ext.lower() in [".jpg", ".jpeg"]:

        ok, buf = cv2.imencode(
            ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

    else:

        ok, buf = cv2.imencode(".png", image_bgr)

    if not ok:

        return None

    return buf.tobytes()


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
# SIDEBAR: PARAMETER DETEKSI
# ------------------------------------------------------------

with st.sidebar:

    st.markdown("### ⚙️ Parameter Deteksi")

    st.caption(
        "Default biasanya sudah cukup. Atur ulang kalau hasil "
        "kurang bersih atau malah kehapus bagian yang bukan teks."
    )

    top_pct = st.slider(
        "Mulai area deteksi dari (% tinggi foto)",
        min_value=40,
        max_value=95,
        value=int(DEFAULT_TOP_PCT * 100),
        step=1,
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
    )

    edge_thresh = st.slider(
        "Sensitivitas outline (edge threshold)",
        min_value=5,
        max_value=80,
        value=DEFAULT_EDGE_THRESH,
        step=5,
        help="Turunkan kalau outline teks tipis/tidak terdeteksi. "
        "Naikkan kalau area lain (misal tekstur lantai) ikut terdeteksi.",
    )

    dilate_px = st.slider(
        "Margin aman sekitar teks (px)",
        min_value=1,
        max_value=20,
        value=DEFAULT_DILATE_PX,
        step=1,
    )

    inpaint_radius = st.slider(
        "Radius rekonstruksi latar (inpaint radius)",
        min_value=1,
        max_value=15,
        value=DEFAULT_INPAINT_RADIUS,
        step=1,
    )

    st.markdown("### ✨ Kehalusan Hasil")

    smooth_mode = st.checkbox(
        "Mode Halus (multi-scale + feathered blending)",
        value=DEFAULT_SMOOTH_MODE,
        help="Rekonstruksi dilakukan berlapis dari resolusi kecil ke "
        "besar, lalu digabung dengan transisi lembut di tepi. Hasil "
        "jauh lebih smooth, tapi sedikit lebih lambat diproses. "
        "Matikan untuk kembali ke mode cepat (1 pass).",
    )

    feather_px = DEFAULT_FEATHER_PX

    if smooth_mode:

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


# ------------------------------------------------------------
# PROSES
# ------------------------------------------------------------

st.markdown("### 2️⃣ Proses & Preview")

process_clicked = st.button(
    "🚀 HAPUS TIMESTAMP DARI SEMUA FOTO",
    type="primary",
    use_container_width=True,
)

if process_clicked:

    results = []

    progress_bar = st.progress(0.0)

    for index, (filename, raw_bytes) in enumerate(input_files):

        try:

            pil_image = load_pil_image_fixed_orientation(raw_bytes)

            image_rgb = np.array(pil_image)

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            result_bgr, mask, detected_px = remove_timestamp(
                image_bgr,
                top_pct=top_pct,
                white_thresh=white_thresh,
                edge_thresh=edge_thresh,
                dilate_px=dilate_px,
                inpaint_radius=inpaint_radius,
                smooth_mode=smooth_mode,
                feather_px=feather_px,
            )

            name, ext = os.path.splitext(filename)

            if not ext:
                ext = ".jpg"

            output_name = name + "_clean" + ext

            encoded_bytes = encode_image_bytes(result_bgr, ext)

            results.append(
                {
                    "filename": filename,
                    "output_name": output_name,
                    "original_bgr": image_bgr,
                    "result_bgr": result_bgr,
                    "mask": mask,
                    "detected_px": detected_px,
                    "encoded_bytes": encoded_bytes,
                    "mime": "image/jpeg" if ext.lower() in [".jpg", ".jpeg"] else "image/png",
                }
            )

        except Exception as error:

            st.error(f"❌ Gagal memproses `{filename}`: {error}")

        progress_bar.progress((index + 1) / len(input_files))

    st.session_state["remover_results"] = results


# ------------------------------------------------------------
# HASIL
# ------------------------------------------------------------

results = st.session_state.get("remover_results")

if results:

    st.markdown("---")
    st.markdown("### ✨ Hasil")

    zero_detect_count = sum(1 for r in results if r["detected_px"] == 0)

    if zero_detect_count > 0:

        st.warning(
            f"⚠️ {zero_detect_count} dari {len(results)} foto TIDAK terdeteksi "
            "ada timestamp di area yang dicari. Coba longgarkan parameter di "
            "sidebar (misal turunkan '% tinggi foto' atau 'edge threshold')."
        )

    for r in results:

        with st.container(border=True):

            st.markdown(f"**{r['filename']}**")

            col_before, col_after = st.columns(2)

            with col_before:

                st.image(
                    cv2.cvtColor(r["original_bgr"], cv2.COLOR_BGR2RGB),
                    caption="Sebelum",
                    use_container_width=True,
                )

            with col_after:

                st.image(
                    cv2.cvtColor(r["result_bgr"], cv2.COLOR_BGR2RGB),
                    caption="Sesudah",
                    use_container_width=True,
                )

            if show_mask_preview:

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
    "Deteksi berbasis pola piksel (putih terang + outline tajam) di "
    "pita bawah foto -- tanpa AI/API key. Mode Halus memakai "
    "multi-scale inpainting + feathered blending supaya transisi "
    "lebih smooth. Kalau hasil kurang pas, atur parameter di "
    "sidebar lalu proses ulang."
)
