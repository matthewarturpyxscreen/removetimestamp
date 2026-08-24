# ============================================================
# TIMESTAMP REMOVER V6
# CLOUDFlare Workers AI + CLASSIC
# ============================================================
#
# V6 FIX:
#
# 1. Cloudflare image dikirim menggunakan image_b64
# 2. Tidak lagi mengirim image sebagai array RGB
# 3. Mask tetap uint8 array sesuai schema Cloudflare
# 4. AI working resolution dibatasi agar request tidak 413
# 5. Struktur ZIP DIPERTAHANKAN
# 6. Nama file ASLI DIPERTAHANKAN
# 7. Bisa upload:
#       - 1 foto
#       - banyak foto
#       - ZIP
# 8. Output:
#       - download satu foto
#       - download semua sebagai ZIP
#
# Model:
# @cf/runwayml/stable-diffusion-v1-5-inpainting
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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Timestamp Remover V6",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIG
# ============================================================

CLOUDFLARE_MODEL = (
    "@cf/runwayml/stable-diffusion-v1-5-inpainting"
)

# ------------------------------------------------------------
# AI WORKING RESOLUTION
# ------------------------------------------------------------
#
# Semakin besar:
#   + detail lebih bagus
#   - request lebih besar
#   - proses lebih lama
#
# 768 cukup aman untuk V6.
#
# ------------------------------------------------------------

AI_MAX_DIMENSION = 768


# ============================================================
# MASK DETECTION DEFAULT
# ============================================================

DEFAULT_TOP_PCT = 0.72
DEFAULT_LEFT_PCT = 0.0
DEFAULT_RIGHT_PCT = 1.0
DEFAULT_BOTTOM_PCT = 1.0

DEFAULT_WHITE_THRESH = 190
DEFAULT_EDGE_THRESH = 25
DEFAULT_DILATE_PX = 7

MIN_BLOB_AREA = 15


# ============================================================
# CLASSIC DEFAULT
# ============================================================

DEFAULT_INPAINT_RADIUS = 5

DEFAULT_SMOOTH_MODE = True

DEFAULT_SMOOTH_SCALES = (
    0.25,
    0.5,
    1.0
)

DEFAULT_FEATHER_PX = 3


# ============================================================
# CLOUDFLARE PROMPT
# ============================================================

DEFAULT_CF_PROMPT = (
    "Photorealistic restoration of the original photograph. "
    "Remove the masked GPS camera timestamp completely. "
    "Reconstruct only the masked region using the surrounding "
    "background. Continue the exact floor, wall, objects, "
    "textures, perspective, lighting, colors and shadows from "
    "the surrounding image. The repaired area must look like "
    "it was originally photographed without any timestamp. "
    "Do not add anything new. Do not change any unmasked area. "
    "Do not generate text."
)


DEFAULT_CF_NEGATIVE_PROMPT = (
    "text, letters, numbers, GPS, timestamp, date, time, "
    "coordinates, watermark, logo, writing, caption, "
    "new objects, duplicated objects, distorted objects, "
    "blur, smudge, artifacts, unrealistic texture"
)


# ============================================================
# RETRY
# ============================================================

CF_MAX_RETRIES = 3
CF_RETRY_BACKOFF_SEC = 2


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_pil_image_fixed_orientation(
    file_bytes
):

    image = Image.open(
        io.BytesIO(file_bytes)
    )

    image = ImageOps.exif_transpose(
        image
    )

    return image.convert("RGB")


# ============================================================
# DETECT TIMESTAMP MASK
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

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # EDGE
    # ========================================================

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
    ).astype(
        np.uint8
    )

    _, edge_mask = cv2.threshold(
        edge_magnitude,
        edge_thresh,
        255,
        cv2.THRESH_BINARY
    )

    # ========================================================
    # WHITE TEXT
    # ========================================================

    _, white_mask = cv2.threshold(
        gray,
        white_thresh,
        255,
        cv2.THRESH_BINARY
    )

    # ========================================================
    # WHITE + EDGE
    # ========================================================

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

    # ========================================================
    # OUTLINE
    # ========================================================

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

    # ========================================================
    # CLOSE
    # ========================================================

    combined_mask = cv2.morphologyEx(
        combined_mask,
        cv2.MORPH_CLOSE,
        np.ones(
            (9, 9),
            np.uint8
        )
    )

    # ========================================================
    # DILATE
    # ========================================================

    if dilate_px > 0:

        kernel_size = max(
            1,
            int(dilate_px)
        )

        combined_mask = cv2.dilate(
            combined_mask,
            np.ones(
                (
                    kernel_size,
                    kernel_size
                ),
                np.uint8
            ),
            iterations=1
        )

    # ========================================================
    # REMOVE SMALL BLOBS
    # ========================================================

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

    # ========================================================
    # FULL MASK
    # ========================================================

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
# CLASSIC FAST
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

    height, width = (
        image_bgr.shape[:2]
    )

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

    white_thresh=DEFAULT_WHITE_THRESH,

    edge_thresh=DEFAULT_EDGE_THRESH,

    dilate_px=DEFAULT_DILATE_PX,

    inpaint_radius=DEFAULT_INPAINT_RADIUS,

    smooth_mode=DEFAULT_SMOOTH_MODE,

    feather_px=DEFAULT_FEATHER_PX,
):

    mask = detect_timestamp_mask(
        image_bgr,

        top_pct=top_pct,

        white_thresh=white_thresh,

        edge_thresh=edge_thresh,

        dilate_px=dilate_px,
    )

    detected_px = int(
        np.count_nonzero(
            mask
        )
    )

    if detected_px == 0:

        return (
            image_bgr.copy(),
            mask,
            0
        )

    if smooth_mode:

        result = reconstruct_smooth(
            image_bgr,
            mask,

            inpaint_radius=(
                inpaint_radius
            ),

            feather_px=(
                feather_px
            ),
        )

    else:

        result = reconstruct_fast(
            image_bgr,
            mask,

            inpaint_radius=(
                inpaint_radius
            ),
        )

    return (
        result,
        mask,
        detected_px
    )


# ============================================================
# PREPARE AI IMAGE
# ============================================================

def prepare_ai_image(
    pil_image,
    full_mask,
    max_dimension=AI_MAX_DIMENSION
):

    original_size = (
        pil_image.size
    )

    image = pil_image.copy()

    # ========================================================
    # RESIZE
    # ========================================================

    current_max = max(
        image.size
    )

    if current_max > max_dimension:

        scale = (
            max_dimension /
            current_max
        )

        new_width = max(
            256,
            int(
                image.width *
                scale
            )
        )

        new_height = max(
            256,
            int(
                image.height *
                scale
            )
        )

        image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

    width, height = (
        image.size
    )

    # ========================================================
    # MASK
    # ========================================================

    mask_pil = Image.fromarray(
        full_mask.astype(
            np.uint8
        ),
        mode="L"
    )

    mask_pil = mask_pil.resize(
        (
            width,
            height
        ),
        Image.Resampling.NEAREST
    )

    mask_pil = mask_pil.point(
        lambda p:
        255
        if p > 127
        else 0
    )

    return (
        image,
        mask_pil,
        original_size
    )


# ============================================================
# IMAGE -> BASE64 JPEG
# ============================================================

def image_to_base64_jpeg(
    pil_image,
    quality=82
):

    buffer = io.BytesIO()

    pil_image.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )

    return encoded


# ============================================================
# MASK -> ARRAY
# ============================================================

def mask_to_uint8_array(
    mask_pil
):

    mask = np.array(
        mask_pil,
        dtype=np.uint8
    )

    # IMPORTANT:
    # Flatten menjadi 1D array
    # sesuai schema Cloudflare.
    return mask.flatten().tolist()


# ============================================================
# CLOUDFLARE AI
# ============================================================

def remove_timestamp_cloudflare(
    pil_image,
    full_mask,

    account_id,
    api_token,

    prompt=DEFAULT_CF_PROMPT,

    negative_prompt=(
        DEFAULT_CF_NEGATIVE_PROMPT
    ),

    num_steps=20,

    strength=1.0,

    guidance=7.5,

    image_quality=82,
):

    # ========================================================
    # VALIDATION
    # ========================================================

    if not account_id:

        return (
            None,
            "Cloudflare Account ID kosong."
        )

    if not api_token:

        return (
            None,
            "Cloudflare API Token kosong."
        )

    # ========================================================
    # PREPARE
    # ========================================================

    (
        working_image,
        working_mask,
        original_size
    ) = prepare_ai_image(
        pil_image,
        full_mask,
        max_dimension=(
            AI_MAX_DIMENSION
        )
    )

    width, height = (
        working_image.size
    )

    # ========================================================
    # MAKE BASE64 IMAGE
    # ========================================================

    image_b64 = (
        image_to_base64_jpeg(
            working_image,
            quality=image_quality
        )
    )

    # ========================================================
    # MAKE MASK
    # ========================================================

    mask_array = (
        mask_to_uint8_array(
            working_mask
        )
    )

    expected_mask_length = (
        width * height
    )

    if len(mask_array) != (
        expected_mask_length
    ):

        return (
            None,
            "Mask size tidak cocok "
            f"dengan image: "
            f"{len(mask_array)} vs "
            f"{expected_mask_length}"
        )

    # ========================================================
    # ENDPOINT
    # ========================================================

    endpoint = (
        "https://api.cloudflare.com/"
        "client/v4/accounts/"
        f"{account_id}/ai/run/"
        f"{CLOUDFLARE_MODEL}"
    )

    headers = {
        "Authorization":
            f"Bearer {api_token}",

        "Content-Type":
            "application/json",
    }

    # ========================================================
    # PAYLOAD
    # ========================================================
    #
    # IMPORTANT V6:
    #
    # image_b64
    #   -> Base64 JPEG
    #
    # mask
    #   -> uint8 array
    #
    # Jadi jauh lebih kecil daripada
    # V5 yang mengirim RGB image sebagai
    # array integer.
    #
    # ========================================================

    payload = {

        "prompt":
            prompt,

        "negative_prompt":
            negative_prompt,

        "image_b64":
            image_b64,

        "mask":
            mask_array,

        "width":
            int(width),

        "height":
            int(height),

        "num_steps":
            min(
                20,
                max(
                    1,
                    int(num_steps)
                )
            ),

        "strength":
            min(
                1.0,
                max(
                    0.0,
                    float(strength)
                )
            ),

        "guidance":
            float(guidance),
    }

    # ========================================================
    # DEBUG SIZE
    # ========================================================

    payload_size_mb = (
        len(
            str(payload)
        ) / (
            1024 * 1024
        )
    )

    last_error = None

    # ========================================================
    # REQUEST
    # ========================================================

    for attempt in range(
        1,
        CF_MAX_RETRIES + 1
    ):

        try:

            response = requests.post(
                endpoint,

                headers=headers,

                json=payload,

                timeout=300
            )

            # =================================================
            # HTTP ERROR
            # =================================================

            if response.status_code != 200:

                try:

                    error_data = (
                        response.json()
                    )

                except Exception:

                    error_data = (
                        response.text[
                            :3000
                        ]
                    )

                last_error = (
                    f"HTTP "
                    f"{response.status_code}: "
                    f"{error_data}"
                )

                # ---------------------------------------------
                # SPECIAL 413
                # ---------------------------------------------

                if (
                    response.status_code
                    == 413
                ):

                    return (
                        None,
                        "Request terlalu besar "
                        f"meskipun sudah diperkecil. "
                        f"Estimasi payload: "
                        f"{payload_size_mb:.2f} MB. "
                        "Coba turunkan "
                        "AI_MAX_DIMENSION "
                        "dari 768 ke 512."
                    )

                if attempt < (
                    CF_MAX_RETRIES
                ):

                    time.sleep(
                        CF_RETRY_BACKOFF_SEC
                        * attempt
                    )

                    continue

                return (
                    None,
                    last_error
                )

            # =================================================
            # RESPONSE CONTENT TYPE
            # =================================================

            content_type = (
                response.headers
                .get(
                    "content-type",
                    ""
                )
                .lower()
            )

            # =================================================
            # DIRECT IMAGE
            # =================================================

            if content_type.startswith(
                "image/"
            ):

                result = Image.open(
                    io.BytesIO(
                        response.content
                    )
                ).convert(
                    "RGB"
                )

                # ------------------------------------------------
                # Restore original resolution
                # ------------------------------------------------

                if result.size != (
                    original_size
                ):

                    result = result.resize(
                        original_size,
                        Image.Resampling.LANCZOS
                    )

                return (
                    result,
                    None
                )

            # =================================================
            # JSON RESPONSE
            # =================================================

            try:

                data = response.json()

            except Exception:

                return (
                    None,
                    "Cloudflare mengembalikan "
                    "HTTP 200 tetapi response "
                    "bukan gambar maupun JSON."
                )

            if not data.get(
                "success",
                True
            ):

                return (
                    None,
                    str(data)
                )

            result_data = (
                data.get(
                    "result"
                )
            )

            # =================================================
            # RESULT STRING
            # =================================================

            if isinstance(
                result_data,
                str
            ):

                try:

                    decoded = (
                        base64.b64decode(
                            result_data
                        )
                    )

                    result = Image.open(
                        io.BytesIO(
                            decoded
                        )
                    ).convert(
                        "RGB"
                    )

                    if result.size != (
                        original_size
                    ):

                        result = result.resize(
                            original_size,
                            Image.Resampling.LANCZOS
                        )

                    return (
                        result,
                        None
                    )

                except Exception:
                    pass

            # =================================================
            # RESULT DICT
            # =================================================

            if isinstance(
                result_data,
                dict
            ):

                possible_keys = (
                    "image",
                    "image_b64",
                    "response",
                )

                for key in possible_keys:

                    possible = (
                        result_data.get(
                            key
                        )
                    )

                    if not possible:

                        continue

                    try:

                        if isinstance(
                            possible,
                            str
                        ):

                            decoded = (
                                base64.b64decode(
                                    possible
                                )
                            )

                        else:

                            decoded = (
                                bytes(
                                    possible
                                )
                            )

                        result = Image.open(
                            io.BytesIO(
                                decoded
                            )
                        ).convert(
                            "RGB"
                        )

                        if result.size != (
                            original_size
                        ):

                            result = result.resize(
                                original_size,
                                Image.Resampling.LANCZOS
                            )

                        return (
                            result,
                            None
                        )

                    except Exception:

                        continue

            return (
                None,
                "Cloudflare HTTP 200 tetapi "
                "gambar hasil tidak ditemukan "
                "di response.\n\n"
                f"Response:\n{data}"
            )

        except requests.exceptions.Timeout:

            last_error = (
                "Request Cloudflare "
                "timeout."
            )

        except requests.exceptions.RequestException as error:

            last_error = (
                f"Request error: "
                f"{error}"
            )

        except Exception as error:

            last_error = str(
                error
            )

        if attempt < (
            CF_MAX_RETRIES
        ):

            time.sleep(
                CF_RETRY_BACKOFF_SEC
                * attempt
            )

    return (
        None,
        last_error
    )


# ============================================================
# ENCODE PIL
# ============================================================

def encode_pil_image(
    image,
    ext
):

    buffer = io.BytesIO()

    if ext.lower() in (
        ".jpg",
        ".jpeg"
    ):

        image.save(
            buffer,
            format="JPEG",
            quality=95,
            optimize=True
        )

    else:

        image.save(
            buffer,
            format="PNG"
        )

    return buffer.getvalue()


# ============================================================
# ENCODE BGR
# ============================================================

def encode_bgr_image(
    image_bgr,
    ext
):

    if ext.lower() in (
        ".jpg",
        ".jpeg"
    ):

        ok, buffer = cv2.imencode(
            ".jpg",
            image_bgr,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

    else:

        ok, buffer = cv2.imencode(
            ".png",
            image_bgr
        )

    if not ok:

        raise ValueError(
            "Gagal encode gambar."
        )

    return buffer.tobytes()


# ============================================================
# COLLECT INPUT FILES
# ============================================================

def collect_input_files(
    uploaded_files
):

    collected = []

    for uploaded in uploaded_files:

        filename = (
            uploaded.name
        )

        # ====================================================
        # ZIP
        # ====================================================

        if filename.lower().endswith(
            ".zip"
        ):

            try:

                with zipfile.ZipFile(
                    uploaded
                ) as archive:

                    for info in (
                        archive.infolist()
                    ):

                        if info.is_dir():

                            continue

                        inner_path = (
                            info.filename
                        )

                        # ------------------------------------
                        # Ignore MacOS
                        # ------------------------------------

                        if (
                            "__MACOSX"
                            in inner_path
                        ):

                            continue

                        # ------------------------------------
                        # Image only
                        # ------------------------------------

                        if not inner_path.lower().endswith(
                            IMAGE_EXTS
                        ):

                            continue

                        with archive.open(
                            info
                        ) as file:

                            raw_bytes = (
                                file.read()
                            )

                        collected.append(
                            {
                                "path":
                                    inner_path,

                                "bytes":
                                    raw_bytes,

                                "source":
                                    "zip",
                            }
                        )

            except zipfile.BadZipFile:

                st.error(
                    f"❌ `{filename}` "
                    "bukan ZIP valid."
                )

        # ====================================================
        # SINGLE IMAGE
        # ====================================================

        elif filename.lower().endswith(
            IMAGE_EXTS
        ):

            collected.append(
                {
                    "path":
                        filename,

                    "bytes":
                        uploaded.getvalue(),

                    "source":
                        "single",
                }
            )

    return collected


# ============================================================
# CREATE ZIP
# ============================================================

def create_output_zip(
    results
):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as archive:

        for result in results:

            archive.writestr(
                result[
                    "output_path"
                ],

                result[
                    "encoded_bytes"
                ]
            )

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 Metode"
    )

    method = st.radio(
        "Pilih metode:",
        [
            "AI - Cloudflare Workers AI",
            "Klasik - OpenCV",
        ],
        index=0
    )

    use_ai = method.startswith(
        "AI"
    )

    # ========================================================
    # CLOUDFLARE
    # ========================================================

    if use_ai:

        st.markdown(
            "### ☁️ Cloudflare"
        )

        account_id = st.text_input(
            "Cloudflare Account ID",
            value=os.environ.get(
                "CLOUDFLARE_ACCOUNT_ID",
                ""
            ),
            type="password"
        )

        api_token = st.text_input(
            "Cloudflare API Token",
            value=os.environ.get(
                "CLOUDFLARE_API_TOKEN",
                ""
            ),
            type="password"
        )

        st.caption(
            f"Model: `{CLOUDFLARE_MODEL}`"
        )

        st.markdown(
            "### ⚙️ AI"
        )

        ai_steps = st.slider(
            "Diffusion Steps",
            5,
            20,
            20,
            1
        )

        ai_strength = st.slider(
            "Strength",
            0.0,
            1.0,
            1.0,
            0.05
        )

        ai_guidance = st.slider(
            "Guidance",
            1.0,
            15.0,
            7.5,
            0.5
        )

        ai_quality = st.slider(
            "AI JPEG Quality",
            60,
            95,
            82,
            1
        )

        with st.expander(
            "📝 AI Prompt"
        ):

            ai_prompt = st.text_area(
                "Prompt",
                value=(
                    DEFAULT_CF_PROMPT
                ),
                height=180
            )

            ai_negative_prompt = (
                st.text_area(
                    "Negative Prompt",
                    value=(
                        DEFAULT_CF_NEGATIVE_PROMPT
                    ),
                    height=150
                )
            )

    else:

        account_id = ""
        api_token = ""

        ai_steps = 20
        ai_strength = 1.0
        ai_guidance = 7.5
        ai_quality = 82

        ai_prompt = (
            DEFAULT_CF_PROMPT
        )

        ai_negative_prompt = (
            DEFAULT_CF_NEGATIVE_PROMPT
        )

    # ========================================================
    # DETECTION
    # ========================================================

    st.markdown(
        "### 🔍 Deteksi Timestamp"
    )

    top_pct = (
        st.slider(
            "Mulai deteksi dari (%)",
            40,
            95,
            int(
                DEFAULT_TOP_PCT * 100
            ),
            1
        )
        / 100.0
    )

    white_thresh = st.slider(
        "White Threshold",
        140,
        250,
        DEFAULT_WHITE_THRESH,
        5
    )

    edge_thresh = st.slider(
        "Edge Threshold",
        5,
        80,
        DEFAULT_EDGE_THRESH,
        5
    )

    dilate_px = st.slider(
        "Margin Mask (px)",
        1,
        20,
        DEFAULT_DILATE_PX,
        1
    )

    # ========================================================
    # CLASSIC
    # ========================================================

    st.markdown(
        "### 🛠️ Classic"
    )

    inpaint_radius = st.slider(
        "Inpaint Radius",
        1,
        15,
        DEFAULT_INPAINT_RADIUS,
        1
    )

    smooth_mode = st.checkbox(
        "Mode Halus",
        value=DEFAULT_SMOOTH_MODE
    )

    feather_px = st.slider(
        "Feather",
        0,
        10,
        DEFAULT_FEATHER_PX,
        1
    )

    show_mask = st.checkbox(
        "Tampilkan Mask",
        value=True
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧹 Timestamp Remover V6"
)

st.caption(
    "Hapus timestamp GPS Camera "
    "menggunakan Cloudflare Workers AI "
    "atau OpenCV."
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "## 1️⃣ Upload Foto / ZIP"
)

uploaded_files = st.file_uploader(
    "Upload JPG / JPEG / PNG atau ZIP",
    type=[
        "jpg",
        "jpeg",
        "png",
        "zip"
    ],
    accept_multiple_files=True,
    help=(
        "Bisa upload satu foto, "
        "banyak foto, atau ZIP."
    )
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
        "⚠️ Tidak ada foto ditemukan."
    )

    st.stop()


st.success(
    f"✅ {len(input_files)} foto siap diproses."
)


# ============================================================
# CLOUDFLARE VALIDATION
# ============================================================

if use_ai:

    if not account_id:

        st.warning(
            "⚠️ Cloudflare Account ID belum diisi."
        )

    if not api_token:

        st.warning(
            "⚠️ Cloudflare API Token belum diisi."
        )


# ============================================================
# PROCESS
# ============================================================

st.markdown(
    "## 2️⃣ Proses"
)

process_disabled = (
    use_ai
    and (
        not account_id
        or not api_token
    )
)

process_clicked = st.button(
    "🚀 HAPUS TIMESTAMP",
    type="primary",
    use_container_width=True,
    disabled=process_disabled
)


# ============================================================
# RUN
# ============================================================

if process_clicked:

    results = []

    progress = st.progress(
        0.0
    )

    status = st.empty()

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

        status.info(
            f"⏳ Memproses "
            f"`{input_path}` "
            f"({index + 1}/{total})"
        )

        try:

            # =================================================
            # LOAD
            # =================================================

            pil_image = (
                load_pil_image_fixed_orientation(
                    raw_bytes
                )
            )

            image_rgb = np.array(
                pil_image
            )

            image_bgr = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR
            )

            # =================================================
            # DETECT MASK
            # =================================================

            mask = detect_timestamp_mask(
                image_bgr,

                top_pct=top_pct,

                white_thresh=(
                    white_thresh
                ),

                edge_thresh=(
                    edge_thresh
                ),

                dilate_px=(
                    dilate_px
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
                    f"terdeteksi: "
                    f"`{input_path}`"
                )

                progress.progress(
                    (index + 1)
                    / total
                )

                continue

            # =================================================
            # EXTENSION
            # =================================================

            name, ext = os.path.splitext(
                input_path
            )

            if not ext:

                ext = ".jpg"

            # =================================================
            # AI
            # =================================================

            if use_ai:

                result_pil, error = (
                    remove_timestamp_cloudflare(
                        pil_image=(
                            pil_image
                        ),

                        full_mask=(
                            mask
                        ),

                        account_id=(
                            account_id
                        ),

                        api_token=(
                            api_token
                        ),

                        prompt=(
                            ai_prompt
                        ),

                        negative_prompt=(
                            ai_negative_prompt
                        ),

                        num_steps=(
                            ai_steps
                        ),

                        strength=(
                            ai_strength
                        ),

                        guidance=(
                            ai_guidance
                        ),

                        image_quality=(
                            ai_quality
                        ),
                    )
                )

                if error:

                    st.error(
                        f"❌ Cloudflare gagal "
                        f"`{input_path}`:\n\n"
                        f"{error}"
                    )

                    progress.progress(
                        (index + 1)
                        / total
                    )

                    continue

                result_rgb = np.array(
                    result_pil
                )

                encoded_bytes = (
                    encode_pil_image(
                        result_pil,
                        ext
                    )
                )

                method_name = (
                    "Cloudflare AI"
                )

            # =================================================
            # CLASSIC
            # =================================================

            else:

                (
                    result_bgr,
                    mask,
                    detected_px
                ) = remove_timestamp_classic(
                    image_bgr,

                    top_pct=top_pct,

                    white_thresh=(
                        white_thresh
                    ),

                    edge_thresh=(
                        edge_thresh
                    ),

                    dilate_px=(
                        dilate_px
                    ),

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

                result_rgb = cv2.cvtColor(
                    result_bgr,
                    cv2.COLOR_BGR2RGB
                )

                encoded_bytes = (
                    encode_bgr_image(
                        result_bgr,
                        ext
                    )
                )

                method_name = (
                    "Classic OpenCV"
                )

            # =================================================
            # RESULT
            # =================================================
            #
            # IMPORTANT:
            #
            # input:
            #
            # folderA/foto001.jpg
            #
            # output:
            #
            # folderA/foto001.jpg
            #
            # Jadi folder + nama file
            # tetap sama.
            #
            # =================================================

            results.append(
                {
                    "filename":
                        os.path.basename(
                            input_path
                        ),

                    "input_path":
                        input_path,

                    "output_path":
                        input_path,

                    "original_rgb":
                        image_rgb,

                    "result_rgb":
                        result_rgb,

                    "mask":
                        mask,

                    "detected_px":
                        detected_px,

                    "encoded_bytes":
                        encoded_bytes,

                    "mime":
                        (
                            "image/jpeg"
                            if ext.lower()
                            in (
                                ".jpg",
                                ".jpeg"
                            )
                            else
                            "image/png"
                        ),

                    "method":
                        method_name,
                }
            )

        except Exception as error:

            st.error(
                f"❌ Error "
                f"`{input_path}`:\n\n"
                f"{error}"
            )

        progress.progress(
            (index + 1) / total
        )

    status.empty()

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
        f"✅ {len(results)} foto berhasil diproses."
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
                f"Path: "
                f"`{result['input_path']}`"
            )

            col_before, col_after = (
                st.columns(2)
            )

            # ------------------------------------------------
            # BEFORE
            # ------------------------------------------------

            with col_before:

                st.image(
                    result[
                        "original_rgb"
                    ],

                    caption="Sebelum",

                    use_container_width=True
                )

            # ------------------------------------------------
            # AFTER
            # ------------------------------------------------

            with col_after:

                st.image(
                    result[
                        "result_rgb"
                    ],

                    caption="Sesudah",

                    use_container_width=True
                )

            st.caption(
                f"Metode: "
                f"`{result['method']}` | "
                f"Mask: "
                f"{result['detected_px']:,} px"
            )

            # ------------------------------------------------
            # MASK
            # ------------------------------------------------

            if show_mask:

                with st.expander(
                    "🔍 Lihat Mask"
                ):

                    st.image(
                        result[
                            "mask"
                        ],

                        caption=(
                            "Area timestamp "
                            "yang dihapus"
                        ),

                        use_container_width=True
                    )

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            st.download_button(
                label=(
                    "⬇️ Download "
                    f"{result['filename']}"
                ),

                data=result[
                    "encoded_bytes"
                ],

                file_name=result[
                    "filename"
                ],

                mime=result[
                    "mime"
                ],

                key=(
                    f"download_{index}"
                )
            )

    # ========================================================
    # ZIP
    # ========================================================

    st.markdown(
        "---"
    )

    st.markdown(
        "## 📦 Download Semua"
    )

    output_zip = (
        create_output_zip(
            results
        )
    )

    st.download_button(
        label=(
            f"⬇️ DOWNLOAD "
            f"{len(results)} FOTO "
            f"SEBAGAI ZIP"
        ),

        data=output_zip,

        file_name=(
            "foto_tanpa_timestamp.zip"
        ),

        mime="application/zip",

        type="primary",

        use_container_width=True
    )

    st.success(
        "📦 Struktur folder dan "
        "nama file asli dipertahankan."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "Timestamp Remover V6 | "
    "Cloudflare Workers AI "
    "Stable Diffusion Inpainting"
)
