# ============================================================
# TIMESTAMP REMOVER V5
# CLOUDFLARE WORKERS AI + CLASSIC
# ============================================================
#
# V5
# ------------------------------------------------------------
# - Remove timestamp GPS Camera
# - Cloudflare Workers AI REST API
# - stable-diffusion-v1-5-inpainting
# - image + mask dikirim sebagai uint8 array
# - Upload single photo
# - Upload multiple photos
# - Upload ZIP
# - Struktur folder ZIP dipertahankan
# - Nama file asli dipertahankan
# - Output ZIP
# - Preview Before / After
# - Classic OpenCV fallback
# - Tidak menggunakan Gemini
#
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
import requests
import streamlit as st

from PIL import Image, ImageOps


# ============================================================
# CONFIG
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
# CLOUDFLARE
# ============================================================

CLOUDFLARE_MODEL = (
    "@cf/runwayml/stable-diffusion-v1-5-inpainting"
)

DEFAULT_CF_PROMPT = (
    "Photorealistic continuation of the original photograph. "
    "Reconstruct only the masked area naturally using the "
    "surrounding background. Preserve the exact original "
    "objects, floor, wall, furniture, lighting, shadows, "
    "perspective, colors, textures and photographic appearance. "
    "The repaired area must blend seamlessly with the surrounding "
    "pixels. Do not add or modify any object outside the masked "
    "area. Do not generate text."
)

DEFAULT_CF_NEGATIVE_PROMPT = (
    "text, letters, numbers, timestamp, GPS, coordinates, "
    "date, time, watermark, logo, writing, sign, caption, "
    "blur, smudge, artifacts, duplicated objects, distorted "
    "objects, new objects, unrealistic texture"
)

CF_MAX_RETRIES = 3
CF_RETRY_BACKOFF_SEC = 2


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Timestamp Remover V5",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_pil_image_fixed_orientation(file_bytes):

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

    top = int(height * top_pct)
    bottom = int(height * bottom_pct)
    left = int(width * left_pct)
    right = int(width * right_pct)

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
    # CONNECT WHITE TEXT + EDGE
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
    # OUTLINE
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
    # CLOSE
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
    # FULL MASK
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
            (small_w, small_h),
            interpolation=cv2.INTER_AREA
        )

        mask_small = cv2.resize(
            mask,
            (small_w, small_h),
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

        method = (
            cv2.INPAINT_NS
            if scale < 1.0
            else cv2.INPAINT_TELEA
        )

        inpainted_small = cv2.inpaint(
            img_small,
            mask_small,
            inpaint_radius,
            method
        )

        inpainted_up = cv2.resize(
            inpainted_small,
            (width, height),
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
    smooth_mode=True,
    feather_px=DEFAULT_FEATHER_PX,
):

    mask = detect_timestamp_mask(
        image_bgr=image_bgr,
        top_pct=top_pct,
        white_thresh=white_thresh,
        edge_thresh=edge_thresh,
        dilate_px=dilate_px,
    )

    detected_px = int(
        np.count_nonzero(mask)
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
            inpaint_radius=inpaint_radius,
            feather_px=feather_px
        )

    else:

        result = reconstruct_fast(
            image_bgr,
            mask,
            inpaint_radius=inpaint_radius
        )

    return (
        result,
        mask,
        detected_px
    )


# ============================================================
# RESIZE FOR CLOUDFLARE
# ============================================================

def prepare_ai_image(
    pil_image,
    mask,
    max_dimension=1024
):

    original_size = pil_image.size

    image = pil_image.copy()

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    if max(image.size) > max_dimension:

        scale = (
            max_dimension /
            max(image.size)
        )

        new_width = max(
            256,
            int(image.width * scale)
        )

        new_height = max(
            256,
            int(image.height * scale)
        )

        image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

    # --------------------------------------------------------
    # Ensure dimensions valid
    # --------------------------------------------------------

    width, height = image.size

    width = min(
        2048,
        max(256, width)
    )

    height = min(
        2048,
        max(256, height)
    )

    if image.size != (
        width,
        height
    ):

        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS
        )

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask_pil = Image.fromarray(
        mask.astype(np.uint8),
        mode="L"
    )

    mask_pil = mask_pil.resize(
        (width, height),
        Image.Resampling.NEAREST
    )

    mask_pil = mask_pil.point(
        lambda p:
        255 if p > 127 else 0
    )

    return (
        image,
        mask_pil,
        original_size
    )


# ============================================================
# PIL -> UINT8 ARRAY
# ============================================================

def image_to_uint8_array(
    pil_image
):

    rgb = np.array(
        pil_image.convert("RGB"),
        dtype=np.uint8
    )

    return rgb.flatten().tolist()


def mask_to_uint8_array(
    mask_pil
):

    mask = np.array(
        mask_pil.convert("L"),
        dtype=np.uint8
    )

    return mask.flatten().tolist()


# ============================================================
# CLOUDFLARE INPAINTING
# ============================================================

def call_cloudflare_inpainting(
    pil_image,
    mask,
    account_id,
    api_token,
    prompt,
    negative_prompt,
    num_steps,
    strength,
    guidance,
):

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
        mask
    )

    width, height = (
        working_image.size
    )

    # ========================================================
    # ARRAY
    # ========================================================

    image_array = (
        image_to_uint8_array(
            working_image
        )
    )

    mask_array = (
        mask_to_uint8_array(
            working_mask
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    expected_pixels = (
        width * height
    )

    expected_image_values = (
        expected_pixels * 3
    )

    if len(image_array) != (
        expected_image_values
    ):

        return (
            None,
            "Ukuran image array tidak "
            "sesuai dengan width/height."
        )

    if len(mask_array) != (
        expected_pixels
    ):

        return (
            None,
            "Ukuran mask array tidak "
            "sesuai dengan width/height."
        )

    # ========================================================
    # ENDPOINT
    # ========================================================

    url = (
        "https://api.cloudflare.com/client/v4/"
        f"accounts/{account_id}/ai/run/"
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
    # IMPORTANT:
    # Cloudflare schema:
    #
    # image = uint8 array
    # mask  = uint8 array
    #
    # BUKAN mask_b64.
    #
    # ========================================================

    payload = {
        "prompt": prompt,

        "negative_prompt":
            negative_prompt,

        "image":
            image_array,

        "mask":
            mask_array,

        "width":
            width,

        "height":
            height,

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
    # REQUEST
    # ========================================================

    last_error = None

    for attempt in range(
        1,
        CF_MAX_RETRIES + 1
    ):

        try:

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=300
            )

            # =================================================
            # HTTP ERROR
            # =================================================

            if response.status_code != 200:

                try:

                    data = response.json()

                except Exception:

                    data = response.text[
                        :2000
                    ]

                last_error = (
                    f"HTTP "
                    f"{response.status_code}: "
                    f"{data}"
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
            # IMAGE RESPONSE
            # =================================================

            content_type = (
                response.headers
                .get(
                    "content-type",
                    ""
                )
                .lower()
            )

            if content_type.startswith(
                "image/"
            ):

                result = Image.open(
                    io.BytesIO(
                        response.content
                    )
                ).convert("RGB")

                if result.size != (
                    width,
                    height
                ):

                    result = result.resize(
                        (
                            width,
                            height
                        ),
                        Image.Resampling.LANCZOS
                    )

                # ---------------------------------------------
                # BACK TO ORIGINAL SIZE
                # ---------------------------------------------

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
                    "Cloudflare HTTP 200 "
                    "tetapi response bukan "
                    "image atau JSON."
                )

            if not data.get(
                "success",
                True
            ):

                return (
                    None,
                    str(data)
                )

            result_data = data.get(
                "result"
            )

            # -------------------------------------------------
            # RESULT STRING
            # -------------------------------------------------

            if isinstance(
                result_data,
                str
            ):

                try:

                    import base64

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

            # -------------------------------------------------
            # RESULT DICT
            # -------------------------------------------------

            if isinstance(
                result_data,
                dict
            ):

                possible = (
                    result_data.get(
                        "image"
                    )
                    or
                    result_data.get(
                        "image_b64"
                    )
                    or
                    result_data.get(
                        "response"
                    )
                )

                if possible:

                    try:

                        import base64

                        decoded = (
                            base64.b64decode(
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

            return (
                None,
                "Cloudflare berhasil "
                "memproses request tetapi "
                "format hasil gambar tidak "
                "dikenali.\n\n"
                f"{data}"
            )

        except requests.exceptions.Timeout:

            last_error = (
                "Request ke Cloudflare "
                "timeout."
            )

        except requests.exceptions.RequestException as error:

            last_error = (
                f"Request error: {error}"
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
# ENCODER
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
            "Gagal encode image."
        )

    return buffer.tobytes()


# ============================================================
# FILE EXTENSIONS
# ============================================================

IMAGE_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
)


# ============================================================
# COLLECT FILES
# ============================================================

def collect_input_files(
    uploaded_files
):

    files = []

    for uploaded in uploaded_files:

        filename = uploaded.name

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

                        path = info.filename

                        if (
                            "__MACOSX"
                            in path
                        ):
                            continue

                        if not path.lower().endswith(
                            IMAGE_EXTS
                        ):
                            continue

                        with archive.open(
                            info
                        ) as file:

                            data = file.read()

                        files.append(
                            {
                                "path": path,
                                "bytes": data,
                                "source": "zip",
                            }
                        )

            except zipfile.BadZipFile:

                st.error(
                    f"❌ `{filename}` "
                    "bukan ZIP yang valid."
                )

        # ====================================================
        # IMAGE
        # ====================================================

        elif filename.lower().endswith(
            IMAGE_EXTS
        ):

            files.append(
                {
                    "path": filename,
                    "bytes": uploaded.getvalue(),
                    "source": "single",
                }
            )

    return files


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
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for result in results:

            archive.writestr(
                result["output_path"],
                result["encoded_bytes"]
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
        index=0,
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
            "Account ID",
            value=os.environ.get(
                "CLOUDFLARE_ACCOUNT_ID",
                ""
            ),
            type="password"
        )

        api_token = st.text_input(
            "API Token",
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
            "Steps",
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

        with st.expander(
            "📝 Prompt AI"
        ):

            ai_prompt = st.text_area(
                "Prompt",
                value=DEFAULT_CF_PROMPT,
                height=180
            )

            ai_negative_prompt = (
                st.text_area(
                    "Negative Prompt",
                    value=(
                        DEFAULT_CF_NEGATIVE_PROMPT
                    ),
                    height=140
                )
            )

    else:

        account_id = ""
        api_token = ""

        ai_steps = 20
        ai_strength = 1.0
        ai_guidance = 7.5

        ai_prompt = DEFAULT_CF_PROMPT
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
        DEFAULT_SMOOTH_MODE
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
        True
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧹 Timestamp Remover V5"
)

st.caption(
    "Hapus timestamp GPS Camera secara otomatis "
    "dengan Cloudflare Workers AI atau OpenCV."
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "## 1️⃣ Upload Foto / ZIP"
)

uploaded_files = st.file_uploader(
    "Upload JPG, JPEG, PNG atau ZIP",
    type=[
        "jpg",
        "jpeg",
        "png",
        "zip"
    ],
    accept_multiple_files=True
)

if not uploaded_files:

    st.info(
        "📷 Upload foto atau ZIP untuk mulai."
    )

    st.stop()


input_files = (
    collect_input_files(
        uploaded_files
    )
)

if not input_files:

    st.warning(
        "Tidak ada foto yang ditemukan."
    )

    st.stop()


st.success(
    f"✅ {len(input_files)} foto ditemukan."
)


# ============================================================
# VALIDATION
# ============================================================

if use_ai:

    if not account_id:

        st.warning(
            "⚠️ Isi Cloudflare Account ID."
        )

    if not api_token:

        st.warning(
            "⚠️ Isi Cloudflare API Token."
        )


# ============================================================
# PROCESS BUTTON
# ============================================================

st.markdown(
    "## 2️⃣ Proses"
)

disabled = (
    use_ai
    and (
        not account_id
        or not api_token
    )
)

start = st.button(
    "🚀 HAPUS TIMESTAMP",
    type="primary",
    use_container_width=True,
    disabled=disabled
)


# ============================================================
# PROCESS
# ============================================================

if start:

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

        status.write(
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

            name, ext = os.path.splitext(
                input_path
            )

            if not ext:

                ext = ".jpg"

            # =================================================
            # IMAGE NUMPY
            # =================================================

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
                image_bgr=image_bgr,
                top_pct=top_pct,
                white_thresh=white_thresh,
                edge_thresh=edge_thresh,
                dilate_px=dilate_px
            )

            detected_px = int(
                np.count_nonzero(
                    mask
                )
            )

            if detected_px == 0:

                st.warning(
                    f"⚠️ Timestamp tidak "
                    f"terdeteksi pada "
                    f"`{input_path}`"
                )

                continue

            # =================================================
            # AI
            # =================================================

            if use_ai:

                result_pil, error = (
                    call_cloudflare_inpainting(
                        pil_image=pil_image,
                        mask=mask,
                        account_id=account_id,
                        api_token=api_token,
                        prompt=ai_prompt,
                        negative_prompt=(
                            ai_negative_prompt
                        ),
                        num_steps=ai_steps,
                        strength=ai_strength,
                        guidance=ai_guidance,
                    )
                )

                if error:

                    st.error(
                        f"❌ Cloudflare gagal "
                        f"`{input_path}`:\n\n"
                        f"{error}"
                    )

                    continue

                result_rgb = np.array(
                    result_pil
                )

                encoded = (
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

                result_bgr, mask, detected_px = (
                    remove_timestamp_classic(
                        image_bgr=image_bgr,
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

                result_rgb = cv2.cvtColor(
                    result_bgr,
                    cv2.COLOR_BGR2RGB
                )

                result_pil = Image.fromarray(
                    result_rgb
                )

                encoded = (
                    encode_bgr_image(
                        result_bgr,
                        ext
                    )
                )

                method_name = "Classic"

            # =================================================
            # RESULT
            # =================================================
            #
            # IMPORTANT:
            # output_path = input_path
            #
            # Jadi:
            #
            # Bandung/IMG001.jpg
            #
            # tetap:
            #
            # Bandung/IMG001.jpg
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
                        encoded,

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
                f"❌ Error pada "
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
        f"✅ {len(results)} foto berhasil."
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

            col1, col2 = st.columns(
                2
            )

            with col1:

                st.image(
                    result[
                        "original_rgb"
                    ],
                    caption="Sebelum",
                    use_container_width=True
                )

            with col2:

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

            if show_mask:

                with st.expander(
                    "🔍 Lihat Mask"
                ):

                    st.image(
                        result["mask"],
                        caption=(
                            "Area yang "
                            "dikirim untuk "
                            "inpainting"
                        ),
                        use_container_width=True
                    )

            st.download_button(
                label=(
                    f"⬇️ Download "
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
        "## 📦 ZIP Hasil"
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
        "📦 Struktur folder ZIP "
        "dipertahankan dan nama file "
        "asli tidak diubah."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "Timestamp Remover V5 | "
    "Cloudflare Workers AI "
    "stable-diffusion-v1-5-inpainting"
)
