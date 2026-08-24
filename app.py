# ============================================================
# TIMESTAMP REMOVER V8.0 - STREAMLIT
# ============================================================
#
# ALUR BARU:
# Upload foto
#     ↓
# Gemini Vision mendeteksi area timestamp
#     ↓
# OpenCV membuat pixel mask timestamp
#     ↓
# Mask diperlebar secara aman untuk outline/shadow
#     ↓
# Inpainting OpenCV
#     ↓
# Preview + Download
#
# TIDAK ADA:
# - input text manual
# - OCR timestamp
# - font renderer
# - generate timestamp
#
# ============================================================

import os
import io
import re
import json
import base64
import hashlib
import secrets
import subprocess
import glob

import cv2
import numpy as np
import streamlit as st

from PIL import Image, ImageOps
from google import genai
from google.genai import types as genai_types


# ============================================================
# CONFIG
# ============================================================

GEMINI_MODEL_NAME = "gemini-flash-lite-latest"

# Seberapa jauh mask diperluas untuk menangkap outline/shadow.
MASK_DILATE_PX = 3

# Radius inpainting.
INPAINT_RADIUS = 3

# OpenCV inpainting method.
# TELEA biasanya bagus untuk area teks kecil.
INPAINT_METHOD = cv2.INPAINT_TELEA

# Untuk menghindari mask mengambil seluruh background.
# Pixel putih timestamp biasanya memiliki saturation rendah.
WHITE_VALUE_THRESHOLD = 150
WHITE_SATURATION_THRESHOLD = 90

# Toleransi untuk pixel gelap yang berada di sekitar teks putih.
DARK_VALUE_THRESHOLD = 100

# Timestamp GPS camera umumnya berada di bagian bawah foto.
BOTTOM_REGION_RATIO = 0.45

# Maksimum area mask dibanding luas region timestamp.
# Safety guard agar background tidak ikut terhapus besar-besaran.
MAX_MASK_RATIO_IN_REGION = 0.35


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Timestamp Remover | AI",
    page_icon="🧹",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(140, 140, 140, 0.16);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }

    .hero-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        background: linear-gradient(135deg, #059669, #10b981);
        color: white !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.6rem;
    }

    .hero-desc {
        color: var(--text-color, #6b7280);
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1rem;
    }

    .info-card {
        padding: 12px 16px;
        border-radius: 10px;
        background: rgba(140, 140, 140, 0.06);
        border-left: 4px solid #10b981;
        margin: 10px 0;
        font-size: 0.9rem;
    }

    .footer-text {
        text-align: center;
        font-size: 0.82rem;
        opacity: 0.75;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "uploader_version": 0,
    "remove_result": None,
    "last_detection_signature": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_app():
    st.session_state.uploader_version += 1
    st.session_state.remove_result = None
    st.session_state.last_detection_signature = None


uploader_version = st.session_state.uploader_version


# ============================================================
# LOAD IMAGE
# ============================================================

def load_pil_image_fixed_orientation(uploaded_file):
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


# ============================================================
# GEMINI PROMPT
# ============================================================

GEMINI_REMOVE_PROMPT = """
You are an image analysis system whose ONLY task is to locate a GPS
camera timestamp overlay so it can be removed from the image.

IMPORTANT:
- Do NOT transcribe the timestamp text.
- Do NOT return the timestamp text.
- Do NOT describe the timestamp contents.
- Only identify its visual region.
- The timestamp may contain multiple lines.
- It may contain GPS coordinates, location, date, time, and timezone.
- It is commonly located near the bottom of the image.
- It may use white text with a black outline or shadow.
- Do not include normal objects, people, buildings, roads, sky,
  vegetation, or other photographic content in the timestamp box.

Return ONLY valid JSON in this exact structure:

{
  "found": true,
  "confidence": 0.0,
  "box": {
    "x1": 0,
    "y1": 0,
    "x2": 0,
    "y2": 0
  }
}

Coordinates MUST be normalized from 0 to 1000 relative to the image:
- x1 = left
- y1 = top
- x2 = right
- y2 = bottom

If there is no GPS camera timestamp overlay, return:

{
  "found": false,
  "confidence": 0.0,
  "box": {
    "x1": 0,
    "y1": 0,
    "x2": 0,
    "y2": 0
  }
}
""".strip()


# ============================================================
# GEMINI DETECTION
# ============================================================

def _extract_json_from_response(text):
    text = (text or "").strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Coba langsung.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Cari object JSON pertama.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        return json.loads(match.group(0))

    raise ValueError("Gemini tidak mengembalikan JSON yang valid.")


def detect_timestamp_region(pil_image, api_key):
    client = genai.Client(api_key=api_key)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)

    image_bytes = buffer.getvalue()

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=[
            genai_types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            ),
            GEMINI_REMOVE_PROMPT,
        ],
    )

    data = _extract_json_from_response(response.text)

    found = bool(data.get("found", False))
    confidence = float(data.get("confidence", 0.0) or 0.0)

    box = data.get("box") or {}

    x1 = float(box.get("x1", 0))
    y1 = float(box.get("y1", 0))
    x2 = float(box.get("x2", 0))
    y2 = float(box.get("y2", 0))

    return {
        "found": found,
        "confidence": max(0.0, min(1.0, confidence)),
        "box_norm": {
            "x1": max(0.0, min(1000.0, x1)),
            "y1": max(0.0, min(1000.0, y1)),
            "x2": max(0.0, min(1000.0, x2)),
            "y2": max(0.0, min(1000.0, y2)),
        },
    }


# ============================================================
# BOX CONVERSION
# ============================================================

def normalized_box_to_pixels(box_norm, width, height):
    x1 = int(round((box_norm["x1"] / 1000.0) * width))
    y1 = int(round((box_norm["y1"] / 1000.0) * height))
    x2 = int(round((box_norm["x2"] / 1000.0) * width))
    y2 = int(round((box_norm["y2"] / 1000.0) * height))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    return x1, y1, x2, y2


def validate_box(x1, y1, x2, y2, width, height):
    if x2 <= x1 or y2 <= y1:
        return False

    box_area = (x2 - x1) * (y2 - y1)

    if box_area <= 0:
        return False

    image_area = width * height

    # Timestamp box yang tiba-tiba mengambil > 60% foto
    # dianggap tidak valid.
    if box_area > image_area * 0.60:
        return False

    return True


# ============================================================
# PIXEL MASK
# ============================================================

def create_timestamp_pixel_mask(image_bgr, box):
    """
    Membuat mask dari pixel timestamp di dalam bounding box AI.

    Strategi:
    1. Fokus hanya pada region yang diberikan Gemini.
    2. Cari pixel terang/putih yang umum digunakan timestamp.
    3. Cari edge/outline yang dekat dengan pixel timestamp.
    4. Gunakan morphology untuk menghubungkan karakter.
    5. Batasi luas mask sebagai safety guard.
    """

    height, width = image_bgr.shape[:2]

    x1, y1, x2, y2 = box

    region = image_bgr[y1:y2, x1:x2].copy()

    if region.size == 0:
        return np.zeros((height, width), dtype=np.uint8), {
            "mask_pixels": 0,
            "mask_ratio": 0.0,
            "box": box,
        }

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # --------------------------------------------------------
    # MASK 1: PIXEL PUTIH / TERANG
    # --------------------------------------------------------

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, WHITE_VALUE_THRESHOLD], dtype=np.uint8),
        np.array([180, WHITE_SATURATION_THRESHOLD, 255], dtype=np.uint8),
    )

    # --------------------------------------------------------
    # MASK 2: EDGE
    # --------------------------------------------------------

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    # Edge hanya dipakai di sekitar kandidat pixel putih.
    nearby_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    expanded_white = cv2.dilate(
        white_mask,
        nearby_kernel,
        iterations=1,
    )

    nearby_edges = cv2.bitwise_and(
        edges,
        expanded_white,
    )

    # --------------------------------------------------------
    # GABUNG
    # --------------------------------------------------------

    mask_region = cv2.bitwise_or(
        white_mask,
        nearby_edges,
    )

    # --------------------------------------------------------
    # CLEAN NOISE
    # --------------------------------------------------------

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2),
    )

    mask_region = cv2.morphologyEx(
        mask_region,
        cv2.MORPH_OPEN,
        small_kernel,
        iterations=1,
    )

    # Sambungkan bagian karakter yang terputus.
    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    mask_region = cv2.morphologyEx(
        mask_region,
        cv2.MORPH_CLOSE,
        connect_kernel,
        iterations=1,
    )

    # Sedikit dilate untuk menangkap outline hitam.
    if MASK_DILATE_PX > 0:
        dilate_size = MASK_DILATE_PX * 2 + 1

        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_size, dilate_size),
        )

        mask_region = cv2.dilate(
            mask_region,
            dilate_kernel,
            iterations=1,
        )

    # --------------------------------------------------------
    # COMPONENT FILTER
    # --------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_region,
        connectivity=8,
    )

    filtered = np.zeros_like(mask_region)

    region_area = mask_region.shape[0] * mask_region.shape[1]

    min_component_area = max(
        2,
        int(region_area * 0.000002),
    )

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_component_area:
            filtered[labels == label] = 255

    mask_region = filtered

    # --------------------------------------------------------
    # SAFETY GUARD
    # --------------------------------------------------------

    mask_pixels = int(np.count_nonzero(mask_region))
    mask_ratio = mask_pixels / max(1, region_area)

    # Kalau mask terlalu besar, jangan langsung inpaint.
    # Ini biasanya berarti background ikut terdeteksi.
    if mask_ratio > MAX_MASK_RATIO_IN_REGION:
        # Turunkan kembali ke kandidat putih saja.
        mask_region = white_mask.copy()

        mask_region = cv2.morphologyEx(
            mask_region,
            cv2.MORPH_CLOSE,
            connect_kernel,
            iterations=1,
        )

        if MASK_DILATE_PX > 0:
            mask_region = cv2.dilate(
                mask_region,
                dilate_kernel,
                iterations=1,
            )

        mask_pixels = int(np.count_nonzero(mask_region))
        mask_ratio = mask_pixels / max(1, region_area)

    # --------------------------------------------------------
    # MASUKKAN KEMBALI KE UKURAN FOTO
    # --------------------------------------------------------

    full_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    full_mask[y1:y2, x1:x2] = mask_region

    info = {
        "mask_pixels": mask_pixels,
        "mask_ratio": mask_ratio,
        "box": box,
    }

    return full_mask, info


# ============================================================
# OPTIONAL: VISUALIZE MASK
# ============================================================

def create_mask_preview(image_rgb, mask):
    preview = image_rgb.copy()

    # Overlay merah untuk area yang akan diinpaint.
    overlay = preview.copy()
    overlay[mask > 0] = [255, 0, 0]

    preview = cv2.addWeighted(
        preview,
        0.70,
        overlay,
        0.30,
        0,
    )

    return preview


# ============================================================
# INPAINT
# ============================================================

def remove_timestamp_with_inpainting(image_rgb, mask):
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    result_bgr = cv2.inpaint(
        image_bgr,
        mask,
        INPAINT_RADIUS,
        INPAINT_METHOD,
    )

    result_rgb = cv2.cvtColor(
        result_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return result_rgb


# ============================================================
# ENCODE OUTPUT
# ============================================================

def encode_image(image_rgb, original_name):
    name, ext = os.path.splitext(original_name)

    if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    random_word = (
        secrets.choice(
            [
                "bersih",
                "jernih",
                "clean",
                "rapi",
                "natural",
                "fresh",
            ]
        )
        + secrets.choice(
            [
                "foto",
                "image",
                "hasil",
                "camera",
                "photo",
            ]
        )
    )

    output_name = f"{name}_removed_{random_word}{ext}"

    if ext.lower() in [".jpg", ".jpeg"]:
        encode_success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR,
            ),
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        mime_type = "image/jpeg"

    elif ext.lower() == ".webp":
        encode_success, encoded = cv2.imencode(
            ".webp",
            cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR,
            ),
            [
                cv2.IMWRITE_WEBP_QUALITY,
                95,
            ],
        )

        mime_type = "image/webp"

    else:
        encode_success, encoded = cv2.imencode(
            ".png",
            cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR,
            ),
        )

        mime_type = "image/png"

    if not encode_success:
        raise RuntimeError("Gagal melakukan encode gambar hasil.")

    return (
        output_name,
        encoded.tobytes(),
        mime_type,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-badge">⚡ V8.0 • AI Detection + Pixel Mask + Inpainting</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">🧹 Timestamp Remover</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-desc">'
    'Hapus timestamp GPS Camera dari foto secara otomatis menggunakan '
    '<b>Gemini Vision + OpenCV Pixel Mask + Inpainting</b>. '
    'Tidak perlu memasukkan teks timestamp.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

gemini_api_key = st.secrets.get(
    "GEMINI_API_KEY",
    "",
)

if not gemini_api_key:
    st.warning(
        "⚠️ **GEMINI_API_KEY belum dikonfigurasi.** "
        "Tambahkan `GEMINI_API_KEY` pada Streamlit Secrets."
    )


# ============================================================
# STEP 1 - UPLOAD
# ============================================================

st.markdown("### 1️⃣ Upload Foto")

image_rgb = None
photo_file = None

with st.container(border=True):

    photo_file = st.file_uploader(
        "Pilih foto yang ingin dihapus timestamp-nya",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key=f"photo_uploader_{uploader_version}",
        help="Orientasi EXIF akan dinormalisasi otomatis.",
    )

    if photo_file is not None:

        try:

            photo_pil = load_pil_image_fixed_orientation(
                photo_file
            )

            image_rgb = np.array(
                photo_pil
            )

        except Exception as error:

            st.error(
                f"❌ Foto tidak dapat dibaca: {error}"
            )

            image_rgb = None

        if image_rgb is not None:

            height, width = image_rgb.shape[:2]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📄 File",
                    photo_file.name
                    if len(photo_file.name) <= 22
                    else photo_file.name[:19] + "...",
                )

            with col2:
                st.metric(
                    "📐 Resolusi",
                    f"{width} × {height}",
                )

            with col3:
                aspect = width / height if height else 1.0

                if aspect > 1.1:
                    orientation = "Landscape"
                elif aspect < 0.9:
                    orientation = "Portrait"
                else:
                    orientation = "Square"

                st.metric(
                    "🧭 Orientasi",
                    orientation,
                )

            st.image(
                image_rgb,
                caption="Preview Foto Asli",
                use_container_width=True,
            )

    else:

        st.info(
            "📷 Upload foto yang masih memiliki timestamp GPS Camera."
        )


# ============================================================
# STEP 2 - REMOVE
# ============================================================

st.markdown("### 2️⃣ AI Detect & Remove")

photo_ready = image_rgb is not None
api_ready = bool(gemini_api_key)

with st.container(border=True):

    st.markdown(
        """
        <div class="info-card">
        🤖 Gemini AI akan mencari <b>lokasi timestamp</b>, bukan membaca
        isi timestamp. Setelah itu OpenCV membuat <b>pixel mask</b>
        dan mengisi kembali area tersebut menggunakan <b>inpainting</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    remove_clicked = st.button(
        "🧹 DETECT & REMOVE TIMESTAMP",
        type="primary",
        disabled=not (photo_ready and api_ready),
        use_container_width=True,
    )

    if remove_clicked:

        try:

            image_hash = hashlib.md5(
                image_rgb.tobytes()
            ).hexdigest()

            # ------------------------------------------------
            # AI DETECTION
            # ------------------------------------------------

            with st.spinner(
                "🤖 Gemini sedang mencari lokasi timestamp..."
            ):

                detection = detect_timestamp_region(
                    photo_pil,
                    gemini_api_key,
                )

            if not detection["found"]:

                st.error(
                    "❌ Gemini tidak menemukan timestamp GPS Camera "
                    "pada foto ini."
                )

                st.session_state.last_detection_signature = image_hash

            else:

                width = image_rgb.shape[1]
                height = image_rgb.shape[0]

                box = normalized_box_to_pixels(
                    detection["box_norm"],
                    width,
                    height,
                )

                x1, y1, x2, y2 = box

                if not validate_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    width,
                    height,
                ):
                    raise ValueError(
                        "Bounding box dari AI tidak valid."
                    )

                # --------------------------------------------
                # PIXEL MASK
                # --------------------------------------------

                with st.spinner(
                    "🎯 Menganalisis pixel timestamp..."
                ):

                    image_bgr = cv2.cvtColor(
                        image_rgb,
                        cv2.COLOR_RGB2BGR,
                    )

                    mask, mask_info = create_timestamp_pixel_mask(
                        image_bgr,
                        box,
                    )

                mask_pixels = mask_info["mask_pixels"]

                if mask_pixels <= 0:

                    raise ValueError(
                        "Timestamp terdeteksi oleh AI, "
                        "tetapi pixel mask tidak menemukan pixel timestamp."
                    )

                # --------------------------------------------
                # INPAINT
                # --------------------------------------------

                with st.spinner(
                    "🧹 Menghapus timestamp & memperbaiki background..."
                ):

                    result_rgb = remove_timestamp_with_inpainting(
                        image_rgb,
                        mask,
                    )

                (
                    output_name,
                    encoded_bytes,
                    mime_type,
                ) = encode_image(
                    result_rgb,
                    photo_file.name,
                )

                mask_preview = create_mask_preview(
                    image_rgb,
                    mask,
                )

                st.session_state.remove_result = {
                    "original": image_rgb,
                    "result": result_rgb,
                    "mask_preview": mask_preview,
                    "mask": mask,
                    "detection": detection,
                    "box": box,
                    "mask_info": mask_info,
                    "output_name": output_name,
                    "encoded_bytes": encoded_bytes,
                    "mime_type": mime_type,
                }

                st.session_state.last_detection_signature = image_hash

                st.rerun()

        except Exception as error:

            st.error(
                f"❌ Terjadi kesalahan saat remove timestamp: {error}"
            )


# ============================================================
# STEP 3 - RESULT
# ============================================================

result = st.session_state.remove_result

if result is not None:

    st.markdown("---")

    st.markdown("### 3️⃣ Hasil")

    with st.container(border=True):

        st.success(
            "🎉 **Timestamp berhasil diproses!** "
            "Periksa hasil sebelum download."
        )

        confidence = result["detection"]["confidence"]

        box = result["box"]

        mask_info = result["mask_info"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🤖 AI Confidence",
                f"{confidence * 100:.1f}%",
            )

        with col2:
            st.metric(
                "🎯 Mask Pixels",
                f"{mask_info['mask_pixels']:,}",
            )

        with col3:
            st.metric(
                "📦 Mask Ratio",
                f"{mask_info['mask_ratio'] * 100:.2f}%",
            )

        st.markdown(
            f"""
            <div class="info-card">
            <b>AI Detection Box</b><br>
            X1: {box[0]} &nbsp;|&nbsp;
            Y1: {box[1]} &nbsp;|&nbsp;
            X2: {box[2]} &nbsp;|&nbsp;
            Y2: {box[3]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # BEFORE / MASK / RESULT
        # ----------------------------------------------------

        tab_before, tab_mask, tab_result = st.tabs(
            [
                "📷 Original",
                "🎯 Pixel Mask",
                "✨ Hasil Remove",
            ]
        )

        with tab_before:

            st.image(
                result["original"],
                caption="Foto Original",
                use_container_width=True,
            )

        with tab_mask:

            st.image(
                result["mask_preview"],
                caption="Area Pixel yang Akan Dihapus",
                use_container_width=True,
            )

            st.caption(
                "Area merah adalah pixel yang masuk ke proses inpainting."
            )

        with tab_result:

            st.image(
                result["result"],
                caption="Foto Setelah Timestamp Dihapus",
                use_container_width=True,
            )

        # ----------------------------------------------------
        # DOWNLOAD / RESET
        # ----------------------------------------------------

        col_download, col_reset = st.columns([2, 1])

        with col_download:

            st.download_button(
                label=f"⬇️ DOWNLOAD HASIL ({result['output_name']})",
                data=result["encoded_bytes"],
                file_name=result["output_name"],
                mime=result["mime_type"],
                type="primary",
                use_container_width=True,
            )

        with col_reset:

            if st.button(
                "🔄 Foto Baru",
                type="secondary",
                use_container_width=True,
            ):

                reset_app()
                st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer-text">
        <b>Timestamp Remover V8.0</b><br>
        Gemini Vision • OpenCV Pixel Mask • Inpainting
    </div>
    """,
    unsafe_allow_html=True,
)
