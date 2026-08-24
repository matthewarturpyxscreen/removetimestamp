# ============================================================
# TIMESTAMP REMOVER V4
# CLOUDflare Workers AI + CLASSIC
# ============================================================
#
# FITUR
# ------------------------------------------------------------
# 1. Upload satu / banyak foto
# 2. Upload ZIP
# 3. Struktur folder ZIP dipertahankan
# 4. Nama file asli dipertahankan
# 5. Mode Classic:
#       - Deteksi timestamp
#       - OpenCV Inpainting
#
# 6. Mode AI:
#       - Deteksi timestamp
#       - Buat mask
#       - Cloudflare Workers AI
#       - Stable Diffusion v1.5 Inpainting
#
# 7. Tidak menggunakan Gemini
# 8. Tidak membutuhkan google-genai
#
# ============================================================


# ============================================================
# IMPORT
# ============================================================

import io
import os
import time
import zipfile
import base64

import cv2
import numpy as np
import requests
import streamlit as st

from PIL import Image, ImageOps


# ============================================================
# KONFIGURASI DEFAULT
# ============================================================

DEFAULT_TOP_PCT = 0.72
DEFAULT_LEFT_PCT = 0.0
DEFAULT_RIGHT_PCT = 1.0
DEFAULT_BOTTOM_PCT = 1.0

DEFAULT_WHITE_THRESH = 190
DEFAULT_EDGE_THRESH = 25
DEFAULT_DILATE_PX = 7
DEFAULT_INPAINT_RADIUS = 5

MIN_BLOB_AREA = 15

DEFAULT_SMOOTH_MODE = True
DEFAULT_SMOOTH_SCALES = (0.25, 0.5, 1.0)
DEFAULT_FEATHER_PX = 3


# ============================================================
# CLOUDFLARE WORKERS AI
# ============================================================

CLOUDFLARE_MODEL = (
    "@cf/runwayml/stable-diffusion-v1-5-inpainting"
)

DEFAULT_CF_PROMPT = (
    "A realistic continuation of the original photograph background. "
    "Naturally reconstruct the area where the camera timestamp overlay "
    "was removed. Preserve the original floor, wall, objects, lighting, "
    "shadows, colors, perspective, texture and photographic details. "
    "Do not add text, numbers, signs, logos or new objects."
)

DEFAULT_CF_NEGATIVE_PROMPT = (
    "text, timestamp, numbers, GPS coordinates, date, time, watermark, "
    "logo, letters, characters, blur, smudge, artifacts, duplicated "
    "objects, distorted objects, unrealistic texture"
)

CF_MAX_RETRIES = 3
CF_RETRY_BACKOFF_SEC = 2.0


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Timestamp Remover",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_pil_image_fixed_orientation(file_bytes):

    pil_image = Image.open(
        io.BytesIO(file_bytes)
    )

    pil_image = ImageOps.exif_transpose(
        pil_image
    )

    pil_image = pil_image.convert(
        "RGB"
    )

    return pil_image


# ============================================================
# TIMESTAMP DETECTOR
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

    top = int(
        height * top_pct
    )

    bottom = int(
        height * bottom_pct
    )

    left = int(
        width * left_pct
    )

    right = int(
        width * right_pct
    )

    roi = image_bgr[
        top:bottom,
        left:right
    ]

    if roi.size == 0:

        return np.zeros(
            (height, width),
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # EDGE
    # --------------------------------------------------------

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_32F,
        ksize=3
    )

    edge_magnitude = np.abs(
        laplacian
    )

    edge_magnitude = cv2.normalize(
        edge_magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, edge_mask = cv2.threshold(
        edge_magnitude,
        edge_thresh,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # WHITE TEXT
    # --------------------------------------------------------

    _, white_mask = cv2.threshold(
        gray,
        white_thresh,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # CONNECT WHITE TEXT WITH EDGE
    # --------------------------------------------------------

    edge_dilated = cv2.dilate(
        edge_mask,
        np.ones(
            (5, 5),
            np.uint8
        ),
        iterations=1
    )

    text_mask = cv2.bitwise_and(
        white_mask,
        edge_dilated
    )

    # --------------------------------------------------------
    # BLACK OUTLINE
    # --------------------------------------------------------

    text_dilated = cv2.dilate(
        text_mask,
        np.ones(
            (9, 9),
            np.uint8
        ),
        iterations=1
    )

    outline_mask = cv2.bitwise_and(
        edge_mask,
        text_dilated
    )

    combined_mask = cv2.bitwise_or(
        text_mask,
        outline_mask
    )

    # --------------------------------------------------------
    # CLOSE GAP
    # --------------------------------------------------------

    combined_mask = cv2.morphologyEx(
        combined_mask,
        cv2.MORPH_CLOSE,
        np.ones(
            (9, 9),
            np.uint8
        )
    )

    # --------------------------------------------------------
    # DILATE
    # --------------------------------------------------------

    if dilate_px > 0:

        kernel_size = max(
            1,
            int(dilate_px)
        )

        combined_mask = cv2.dilate(
            combined_mask,
            np.ones(
                (kernel_size, kernel_size),
                np.uint8
            ),
            iterations=1
        )

    # --------------------------------------------------------
    # REMOVE SMALL BLOBS
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            combined_mask,
            connectivity=8
        )
    )

    clean_mask = np.zeros_like(
        combined_mask
    )

    for label_index in range(
        1,
        num_labels
    ):

        area = stats[
            label_index,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_BLOB_AREA:

            clean_mask[
                labels == label_index
            ] = 255

    # --------------------------------------------------------
    # FULL SIZE MASK
    # --------------------------------------------------------

    full_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    full_mask[
        top:bottom,
        left:right
    ] = clean_mask

    return full_mask


# ============================================================
# CLASSIC INPAINT
# ============================================================

def reconstruct_fast(
    image_bgr,
    mask,
    inpaint_radius=DEFAULT_INPAINT_RADIUS
):

    return cv2.inpaint(
        image_bgr,
        mask,
        inpaint_radius,
        cv2.INPAINT_TELEA
    )


# ============================================================
# CLASSIC SMOOTH
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

        small_w = max(
            1,
            int(width * scale)
        )

        small_h = max(
            1,
            int(height * scale)
        )

        img_small = cv2.resize(
            result,
            (
                small_w,
                small_h
            ),
            interpolation=cv2.INTER_AREA
        )

        mask_small = cv2.resize(
            mask,
            (
                small_w,
                small_h
            ),
            interpolation=cv2.INTER_NEAREST
        )

        _, mask_small = cv2.threshold(
            mask_small,
            127,
            255,
            cv2.THRESH_BINARY
        )

        if not np.any(mask_small):

            continue

        if scale < 1.0:

            method = cv2.INPAINT_NS

        else:

            method = cv2.INPAINT_TELEA

        inpainted_small = cv2.inpaint(
            img_small,
            mask_small,
            inpaint_radius,
            method
        )

        inpainted_up = cv2.resize(
            inpainted_small,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_CUBIC
        )

        alpha = (
            mask.astype(
                np.float32
            ) / 255.0
        )

        if feather_px > 0:

            alpha = cv2.GaussianBlur(
                alpha,
                (0, 0),
                sigmaX=feather_px
            )

        alpha = np.clip(
            alpha,
            0.0,
            1.0
        )[..., None]

        result = (
            inpainted_up.astype(
                np.float32
            ) * alpha
            +
            result.astype(
                np.float32
            ) * (1.0 - alpha)
        ).astype(
            np.uint8
        )

    return result


# ============================================================
# CLASSIC MAIN
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
        image_bgr=image_bgr,
        top_pct=top_pct,
        left_pct=left_pct,
        right_pct=right_pct,
        bottom_pct=bottom_pct,
        white_thresh=white_thresh,
        edge_thresh=edge_thresh,
        dilate_px=dilate_px,
    )

    detected_px = int(
        np.count_nonzero(
            full_mask
        )
    )

    if detected_px == 0:

        return (
            image_bgr.copy(),
            full_mask,
            detected_px
        )

    if smooth_mode:

        reconstructed = reconstruct_smooth(
            image_bgr,
            full_mask,
            inpaint_radius=inpaint_radius,
            feather_px=feather_px
        )

    else:

        reconstructed = reconstruct_fast(
            image_bgr,
            full_mask,
            inpaint_radius=inpaint_radius
        )

    return (
        reconstructed,
        full_mask,
        detected_px
    )


# ============================================================
# CLOUDFLARE IMAGE ENCODING
# ============================================================

def pil_to_base64(
    pil_image,
    image_format="PNG"
):

    buffer = io.BytesIO()

    pil_image.save(
        buffer,
        format=image_format
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# CLOUDFLARE AI INPAINTING
# ============================================================

def call_cloudflare_inpainting(
    pil_image,
    mask,
    account_id,
    api_token,
    prompt=DEFAULT_CF_PROMPT,
    negative_prompt=DEFAULT_CF_NEGATIVE_PROMPT,
    num_steps=20,
    strength=1.0,
    guidance=7.5,
):

    if not account_id:

        return (
            None,
            "Cloudflare Account ID belum diisi."
        )

    if not api_token:

        return (
            None,
            "Cloudflare API Token belum diisi."
        )

    original_size = pil_image.size

    # --------------------------------------------------------
    # WORKING IMAGE
    # --------------------------------------------------------

    working_image = pil_image.copy()

    max_dimension = 1024

    if max(
        working_image.size
    ) > max_dimension:

        scale = (
            max_dimension
            /
            max(working_image.size)
        )

        new_size = (
            max(
                1,
                int(
                    working_image.width
                    * scale
                )
            ),
            max(
                1,
                int(
                    working_image.height
                    * scale
                )
            ),
        )

        working_image = working_image.resize(
            new_size,
            Image.Resampling.LANCZOS
        )

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    mask_pil = Image.fromarray(
        mask.astype(np.uint8)
    )

    mask_pil = mask_pil.resize(
        working_image.size,
        Image.Resampling.NEAREST
    )

    mask_pil = mask_pil.point(
        lambda p:
        255 if p > 127 else 0
    )

    # --------------------------------------------------------
    # IMAGE BASE64
    # --------------------------------------------------------

    image_b64 = pil_to_base64(
        working_image,
        "PNG"
    )

    # --------------------------------------------------------
    # MASK BASE64
    #
    # Kita kirim mask sebagai PNG juga.
    # --------------------------------------------------------

    mask_b64 = pil_to_base64(
        mask_pil,
        "PNG"
    )

    # --------------------------------------------------------
    # CLOUDFLARE ENDPOINT
    # --------------------------------------------------------

    url = (
        "https://api.cloudflare.com/client/v4/"
        f"accounts/{account_id}/ai/run/"
        f"{CLOUDFLARE_MODEL}"
    )

    headers = {
        "Authorization": (
            f"Bearer {api_token}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image_b64": image_b64,
        "mask_b64": mask_b64,
        "num_steps": int(num_steps),
        "strength": float(strength),
        "guidance": float(guidance),
    }

    last_error = None

    # ========================================================
    # RETRY
    # ========================================================

    for attempt in range(
        1,
        CF_MAX_RETRIES + 1
    ):

        try:

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=180
            )

            # ------------------------------------------------
            # ERROR
            # ------------------------------------------------

            if response.status_code != 200:

                try:

                    error_json = (
                        response.json()
                    )

                    last_error = (
                        f"HTTP {response.status_code}: "
                        f"{error_json}"
                    )

                except Exception:

                    last_error = (
                        f"HTTP {response.status_code}: "
                        f"{response.text[:1000]}"
                    )

                if attempt < CF_MAX_RETRIES:

                    time.sleep(
                        CF_RETRY_BACKOFF_SEC
                        * attempt
                    )

                    continue

                return (
                    None,
                    last_error
                )

            # ------------------------------------------------
            # RESPONSE IMAGE
            # ------------------------------------------------

            content_type = (
                response.headers.get(
                    "content-type",
                    ""
                ).lower()
            )

            if content_type.startswith(
                "image/"
            ):

                result_image = Image.open(
                    io.BytesIO(
                        response.content
                    )
                ).convert("RGB")

                if (
                    result_image.size
                    != original_size
                ):

                    result_image = (
                        result_image.resize(
                            original_size,
                            Image.Resampling.LANCZOS
                        )
                    )

                return (
                    result_image,
                    None
                )

            # ------------------------------------------------
            # RESPONSE JSON
            # ------------------------------------------------

            try:

                response_json = (
                    response.json()
                )

            except Exception:

                return (
                    None,
                    "Cloudflare mengembalikan "
                    "response bukan image/JSON."
                )

            if not response_json.get(
                "success",
                True
            ):

                return (
                    None,
                    str(response_json)
                )

            result = response_json.get(
                "result"
            )

            # ------------------------------------------------
            # RESULT STRING
            # ------------------------------------------------

            if isinstance(
                result,
                str
            ):

                try:

                    image_bytes = (
                        base64.b64decode(
                            result
                        )
                    )

                    result_image = Image.open(
                        io.BytesIO(
                            image_bytes
                        )
                    ).convert("RGB")

                    if (
                        result_image.size
                        != original_size
                    ):

                        result_image = (
                            result_image.resize(
                                original_size,
                                Image.Resampling.LANCZOS
                            )
                        )

                    return (
                        result_image,
                        None
                    )

                except Exception:
                    pass

            # ------------------------------------------------
            # RESULT OBJECT
            # ------------------------------------------------

            if isinstance(
                result,
                dict
            ):

                possible_data = (
                    result.get(
                        "image"
                    )
                    or result.get(
                        "image_b64"
                    )
                    or result.get(
                        "response"
                    )
                )

                if possible_data:

                    try:

                        image_bytes = (
                            base64.b64decode(
                                possible_data
                            )
                        )

                        result_image = (
                            Image.open(
                                io.BytesIO(
                                    image_bytes
                                )
                            ).convert("RGB")
                        )

                        if (
                            result_image.size
                            != original_size
                        ):

                            result_image = (
                                result_image.resize(
                                    original_size,
                                    Image.Resampling.LANCZOS
                                )
                            )

                        return (
                            result_image,
                            None
                        )

                    except Exception:
                        pass

            return (
                None,
                "Format hasil image Cloudflare "
                "tidak dikenali.\n\n"
                f"Response: {response_json}"
            )

        except Exception as error:

            last_error = str(
                error
            )

            if attempt < CF_MAX_RETRIES:

                time.sleep(
                    CF_RETRY_BACKOFF_SEC
                    * attempt
                )

    return (
        None,
        last_error
    )


# ============================================================
# IMAGE ENCODER
# ============================================================

def encode_image_bytes_from_bgr(
    image_bgr,
    ext
):

    if ext.lower() in [
        ".jpg",
        ".jpeg"
    ]:

        ok, buf = cv2.imencode(
            ".jpg",
            image_bgr,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

    else:

        ok, buf = cv2.imencode(
            ".png",
            image_bgr
        )

    if not ok:

        return None

    return buf.tobytes()


def encode_image_bytes_from_pil(
    pil_image,
    ext
):

    buffer = io.BytesIO()

    if ext.lower() in [
        ".jpg",
        ".jpeg"
    ]:

        pil_image.save(
            buffer,
            format="JPEG",
            quality=95
        )

    else:

        pil_image.save(
            buffer,
            format="PNG"
        )

    return buffer.getvalue()


# ============================================================
# SUPPORTED IMAGE
# ============================================================

IMAGE_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
)


# ============================================================
# COLLECT INPUT FILES
# ============================================================

def collect_input_files(
    uploaded_files
):

    collected = []

    for uploaded in uploaded_files:

        name_lower = (
            uploaded.name.lower()
        )

        # ====================================================
        # ZIP
        # ====================================================

        if name_lower.endswith(
            ".zip"
        ):

            with zipfile.ZipFile(
                uploaded
            ) as zip_file:

                for zip_info in (
                    zip_file.infolist()
                ):

                    if zip_info.is_dir():
                        continue

                    inner_name = (
                        zip_info.filename
                    )

                    if (
                        not inner_name
                        .lower()
                        .endswith(
                            IMAGE_EXTS
                        )
                    ):

                        continue

                    if (
                        "__MACOSX"
                        in inner_name
                    ):

                        continue

                    # ----------------------------------------
                    # JANGAN basename()
                    #
                    # Struktur folder dipertahankan.
                    # ----------------------------------------

                    with zip_file.open(
                        zip_info
                    ) as inner_file:

                        collected.append(
                            {
                                "path": inner_name,
                                "bytes": (
                                    inner_file.read()
                                ),
                                "source_type": "zip",
                            }
                        )

        # ====================================================
        # FOTO LANGSUNG
        # ====================================================

        elif name_lower.endswith(
            IMAGE_EXTS
        ):

            collected.append(
                {
                    "path": uploaded.name,
                    "bytes": (
                        uploaded.getvalue()
                    ),
                    "source_type": "photo",
                }
            )

    return collected


# ============================================================
# ZIP OUTPUT
# ============================================================

def create_output_zip(
    results
):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_out:

        for result in results:

            zip_out.writestr(
                result["output_path"],
                result["encoded_bytes"]
            )

    return zip_buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 Metode"
    )

    method = st.radio(
        "Pilih metode",
        options=[
            "Klasik (Offline)",
            "AI - Cloudflare Workers AI",
        ],
        index=0,
    )

    use_ai = method.startswith(
        "AI"
    )

    # ========================================================
    # CLOUDFLARE SETTINGS
    # ========================================================

    if use_ai:

        st.markdown(
            "### ☁️ Cloudflare"
        )

        cloudflare_account_id = (
            st.text_input(
                "Cloudflare Account ID",
                value=os.environ.get(
                    "CLOUDFLARE_ACCOUNT_ID",
                    ""
                ),
                type="password",
            )
        )

        cloudflare_api_token = (
            st.text_input(
                "Cloudflare API Token",
                value=os.environ.get(
                    "CLOUDFLARE_API_TOKEN",
                    ""
                ),
                type="password",
            )
        )

        st.caption(
            f"Model: `{CLOUDFLARE_MODEL}`"
        )

        st.markdown(
            "### ✨ AI Settings"
        )

        cloudflare_steps = st.slider(
            "AI Steps",
            min_value=10,
            max_value=30,
            value=20,
            step=1,
        )

        cloudflare_strength = st.slider(
            "AI Strength",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.05,
        )

        cloudflare_guidance = st.slider(
            "AI Guidance",
            min_value=1.0,
            max_value=15.0,
            value=7.5,
            step=0.5,
        )

        with st.expander(
            "✏️ Prompt AI"
        ):

            cloudflare_prompt = (
                st.text_area(
                    "Prompt",
                    value=DEFAULT_CF_PROMPT,
                    height=150,
                )
            )

            cloudflare_negative_prompt = (
                st.text_area(
                    "Negative Prompt",
                    value=(
                        DEFAULT_CF_NEGATIVE_PROMPT
                    ),
                    height=120,
                )
            )

    else:

        cloudflare_account_id = None
        cloudflare_api_token = None

        cloudflare_prompt = (
            DEFAULT_CF_PROMPT
        )

        cloudflare_negative_prompt = (
            DEFAULT_CF_NEGATIVE_PROMPT
        )

        cloudflare_steps = 20
        cloudflare_strength = 1.0
        cloudflare_guidance = 7.5

    # ========================================================
    # CLASSIC SETTINGS
    # ========================================================

    st.markdown(
        "### ⚙️ Parameter Deteksi"
    )

    top_pct = (
        st.slider(
            "Mulai area deteksi (% tinggi foto)",
            min_value=40,
            max_value=95,
            value=int(
                DEFAULT_TOP_PCT * 100
            ),
            step=1,
        )
        / 100.0
    )

    white_thresh = st.slider(
        "White Threshold",
        min_value=140,
        max_value=250,
        value=DEFAULT_WHITE_THRESH,
        step=5,
    )

    edge_thresh = st.slider(
        "Edge Threshold",
        min_value=5,
        max_value=80,
        value=DEFAULT_EDGE_THRESH,
        step=5,
    )

    dilate_px = st.slider(
        "Margin Timestamp (px)",
        min_value=1,
        max_value=20,
        value=DEFAULT_DILATE_PX,
        step=1,
    )

    inpaint_radius = st.slider(
        "Classic Inpaint Radius",
        min_value=1,
        max_value=15,
        value=DEFAULT_INPAINT_RADIUS,
        step=1,
    )

    smooth_mode = st.checkbox(
        "Mode Halus Classic",
        value=DEFAULT_SMOOTH_MODE,
    )

    feather_px = DEFAULT_FEATHER_PX

    if smooth_mode:

        feather_px = st.slider(
            "Feather",
            min_value=0,
            max_value=10,
            value=DEFAULT_FEATHER_PX,
            step=1,
        )

    show_mask_preview = st.checkbox(
        "Tampilkan preview mask",
        value=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "# 🧹 Timestamp Remover"
)

st.caption(
    "Hapus timestamp GPS Map Camera dari foto "
    "menggunakan Classic Inpainting atau "
    "Cloudflare Workers AI."
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "## 1️⃣ Upload"
)

uploaded_files = st.file_uploader(
    "Upload JPG / JPEG / PNG atau ZIP",
    type=[
        "jpg",
        "jpeg",
        "png",
        "zip",
    ],
    accept_multiple_files=True,
)

if not uploaded_files:

    st.info(
        "📷 Silakan upload foto atau ZIP."
    )

    st.stop()


# ============================================================
# COLLECT
# ============================================================

input_files = (
    collect_input_files(
        uploaded_files
    )
)

if not input_files:

    st.warning(
        "Tidak ditemukan foto."
    )

    st.stop()


st.success(
    f"✅ {len(input_files)} foto siap diproses."
)


# ============================================================
# VALIDATE CLOUDFLARE
# ============================================================

if use_ai:

    if not cloudflare_account_id:

        st.warning(
            "⚠️ Cloudflare Account ID belum diisi."
        )

    if not cloudflare_api_token:

        st.warning(
            "⚠️ Cloudflare API Token belum diisi."
        )


# ============================================================
# PROCESS BUTTON
# ============================================================

st.markdown(
    "## 2️⃣ Proses"
)

process_disabled = (
    use_ai
    and (
        not cloudflare_account_id
        or not cloudflare_api_token
    )
)

process_clicked = st.button(
    "🚀 HAPUS TIMESTAMP",
    type="primary",
    use_container_width=True,
    disabled=process_disabled,
)


# ============================================================
# PROCESS
# ============================================================

if process_clicked:

    results = []

    progress_bar = st.progress(
        0.0
    )

    status_text = st.empty()

    total = len(
        input_files
    )

    for index, item in enumerate(
        input_files
    ):

        input_path = item[
            "path"
        ]

        raw_bytes = item[
            "bytes"
        ]

        status_text.caption(
            f"Memproses `{input_path}` "
            f"({index + 1}/{total})..."
        )

        try:

            # ----------------------------------------------
            # LOAD
            # ----------------------------------------------

            pil_image = (
                load_pil_image_fixed_orientation(
                    raw_bytes
                )
            )

            name, ext = (
                os.path.splitext(
                    input_path
                )
            )

            if not ext:

                ext = ".jpg"

            # =================================================
            # MODE AI
            # =================================================

            if use_ai:

                image_rgb = np.array(
                    pil_image
                )

                image_bgr = (
                    cv2.cvtColor(
                        image_rgb,
                        cv2.COLOR_RGB2BGR
                    )
                )

                # ----------------------------------------------
                # DETECT MASK
                # ----------------------------------------------

                mask = (
                    detect_timestamp_mask(
                        image_bgr,
                        top_pct=top_pct,
                        white_thresh=white_thresh,
                        edge_thresh=edge_thresh,
                        dilate_px=dilate_px,
                    )
                )

                detected_px = int(
                    np.count_nonzero(
                        mask
                    )
                )

                if detected_px == 0:

                    st.warning(
                        f"⚠️ Timestamp tidak "
                        f"terdeteksi: `{input_path}`"
                    )

                    continue

                # ----------------------------------------------
                # CLOUDFLARE
                # ----------------------------------------------

                result_pil, error = (
                    call_cloudflare_inpainting(
                        pil_image=pil_image,
                        mask=mask,
                        account_id=(
                            cloudflare_account_id
                        ),
                        api_token=(
                            cloudflare_api_token
                        ),
                        prompt=(
                            cloudflare_prompt
                        ),
                        negative_prompt=(
                            cloudflare_negative_prompt
                        ),
                        num_steps=(
                            cloudflare_steps
                        ),
                        strength=(
                            cloudflare_strength
                        ),
                        guidance=(
                            cloudflare_guidance
                        ),
                    )
                )

                if error:

                    st.error(
                        f"❌ Cloudflare gagal "
                        f"`{input_path}`:\n\n"
                        f"{error}"
                    )

                    continue

                # ----------------------------------------------
                # ENCODE
                # ----------------------------------------------

                encoded_bytes = (
                    encode_image_bytes_from_pil(
                        result_pil,
                        ext
                    )
                )

                results.append(
                    {
                        "filename": os.path.basename(
                            input_path
                        ),

                        "input_path": input_path,

                        # Struktur folder dipertahankan
                        "output_path": input_path,

                        "original_rgb": (
                            image_rgb
                        ),

                        "result_rgb": (
                            np.array(
                                result_pil
                            )
                        ),

                        "mask": mask,

                        "detected_px": (
                            detected_px
                        ),

                        "encoded_bytes": (
                            encoded_bytes
                        ),

                        "mime": (
                            "image/jpeg"
                            if ext.lower()
                            in [
                                ".jpg",
                                ".jpeg"
                            ]
                            else
                            "image/png"
                        ),

                        "method": (
                            "Cloudflare AI"
                        ),
                    }
                )

            # =================================================
            # MODE CLASSIC
            # =================================================

            else:

                image_rgb = np.array(
                    pil_image
                )

                image_bgr = (
                    cv2.cvtColor(
                        image_rgb,
                        cv2.COLOR_RGB2BGR
                    )
                )

                result_bgr, mask, detected_px = (
                    remove_timestamp_classic(
                        image_bgr,
                        top_pct=top_pct,
                        white_thresh=white_thresh,
                        edge_thresh=edge_thresh,
                        dilate_px=dilate_px,
                        inpaint_radius=(
                            inpaint_radius
                        ),
                        smooth_mode=(
                            smooth_mode
                        ),
                        feather_px=(
                            feather_px
                        ),
                    )
                )

                encoded_bytes = (
                    encode_image_bytes_from_bgr(
                        result_bgr,
                        ext
                    )
                )

                results.append(
                    {
                        "filename": os.path.basename(
                            input_path
                        ),

                        "input_path": input_path,

                        "output_path": input_path,

                        "original_rgb": (
                            image_rgb
                        ),

                        "result_rgb": (
                            cv2.cvtColor(
                                result_bgr,
                                cv2.COLOR_BGR2RGB
                            )
                        ),

                        "mask": mask,

                        "detected_px": (
                            detected_px
                        ),

                        "encoded_bytes": (
                            encoded_bytes
                        ),

                        "mime": (
                            "image/jpeg"
                            if ext.lower()
                            in [
                                ".jpg",
                                ".jpeg"
                            ]
                            else
                            "image/png"
                        ),

                        "method": (
                            "Classic"
                        ),
                    }
                )

        except Exception as error:

            st.error(
                f"❌ Gagal `{input_path}`:\n\n"
                f"{error}"
            )

        progress_bar.progress(
            (index + 1) / total
        )

    status_text.empty()

    st.session_state[
        "remover_results"
    ] = results


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.get(
    "remover_results"
)


if results:

    st.markdown(
        "---"
    )

    st.markdown(
        "## ✨ Hasil"
    )

    st.success(
        f"Berhasil memproses "
        f"{len(results)} foto."
    )

    # ========================================================
    # PREVIEW
    # ========================================================

    for index, result in enumerate(
        results
    ):

        with st.container(
            border=True
        ):

            st.markdown(
                f"### 📷 "
                f"`{result['filename']}`"
            )

            st.caption(
                f"Path: `{result['input_path']}`"
            )

            col_before, col_after = (
                st.columns(2)
            )

            with col_before:

                st.image(
                    result[
                        "original_rgb"
                    ],
                    caption="Sebelum",
                    use_container_width=True,
                )

            with col_after:

                st.image(
                    result[
                        "result_rgb"
                    ],
                    caption="Sesudah",
                    use_container_width=True,
                )

            if (
                show_mask_preview
                and result["mask"]
                is not None
            ):

                with st.expander(
                    "🔍 Preview Mask"
                ):

                    st.image(
                        result[
                            "mask"
                        ],
                        caption=(
                            f"Area terdeteksi: "
                            f"{result['detected_px']:,} piksel"
                        ),
                        use_container_width=True,
                    )

            st.download_button(
                label=(
                    f"⬇️ Download "
                    f"{result['filename']}"
                ),

                data=result[
                    "encoded_bytes"
                ],

                file_name=(
                    result["filename"]
                ),

                mime=result[
                    "mime"
                ],

                key=(
                    f"download_{index}"
                ),
            )


    # ========================================================
    # ZIP
    # ========================================================

    st.markdown(
        "---"
    )

    st.markdown(
        "## 📦 Download ZIP"
    )

    output_zip = (
        create_output_zip(
            results
        )
    )

    st.download_button(
        label=(
            f"⬇️ DOWNLOAD SEMUA "
            f"({len(results)} FOTO) "
            f"SEBAGAI ZIP"
        ),

        data=output_zip,

        file_name=(
            "foto_tanpa_timestamp.zip"
        ),

        mime="application/zip",

        type="primary",

        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "Timestamp Remover V4 — Classic menggunakan "
    "OpenCV Inpainting. Mode AI menggunakan "
    "Cloudflare Workers AI REST API dengan "
    f"`{CLOUDFLARE_MODEL}`. "
    "Struktur folder ZIP dipertahankan."
)
