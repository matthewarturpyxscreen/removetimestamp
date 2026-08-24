# ============================================================
# TIMESTAMP REMOVER V9.0 - GEMINI NATIVE IMAGE EDITING
# ============================================================
#
# Konsep V9:
#
#   Upload foto / ZIP
#          ↓
#   Gemini Image Editing
#          ↓
#   Gemini sendiri mendeteksi timestamp
#   + menghapusnya
#   + merekonstruksi background
#          ↓
#   Output dengan nama & struktur yang sama
#
# TIDAK menggunakan:
# - Gemini OCR / bounding box
# - OpenCV pixel threshold
# - rectangle mask
# - manual text input
#
# Gemini dipakai sebagai IMAGE EDITOR langsung,
# seperti workflow edit gambar di Gemini.
#
# ============================================================

import io
import os
import base64
import zipfile
from pathlib import Path

import streamlit as st
import numpy as np

from PIL import Image, ImageOps
from google import genai


# ============================================================
# CONFIG
# ============================================================

# Gemini native image editing model.
GEMINI_IMAGE_MODEL = "gemini-flash-lite-latest"

# Ukuran output Gemini.
# 2K dipilih agar hasil edit lebih detail daripada 1K.
GEMINI_IMAGE_SIZE = "2K"

SUPPORTED_IMAGES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Timestamp Remover | Gemini AI",
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

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0

if "remove_result" not in st.session_state:
    st.session_state.remove_result = None


def reset_app():
    st.session_state.uploader_version += 1
    st.session_state.remove_result = None


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_pil_image(uploaded_file):
    """
    Baca foto + normalisasi EXIF orientation.
    """
    image = Image.open(uploaded_file)

    image = ImageOps.exif_transpose(image)

    return image.convert("RGB")


def image_to_bytes(image, extension):
    """
    Encode PIL image ke format output yang sama.
    """
    buffer = io.BytesIO()

    ext = extension.lower()

    if ext in [".jpg", ".jpeg"]:

        image.save(
            buffer,
            format="JPEG",
            quality=95,
            subsampling=0,
        )

    elif ext == ".png":

        image.save(
            buffer,
            format="PNG",
        )

    elif ext == ".webp":

        image.save(
            buffer,
            format="WEBP",
            quality=95,
        )

    else:

        image.save(
            buffer,
            format="JPEG",
            quality=95,
        )

    return buffer.getvalue()


def get_image_from_gemini_response(response):
    """
    Mengambil gambar hasil dari Gemini SDK.

    Mendukung response.parts / inline_data
    dari generate_content.
    """

    # Cara utama untuk google-genai generate_content.
    for part in getattr(response, "parts", []) or []:

        inline_data = getattr(
            part,
            "inline_data",
            None,
        )

        if inline_data is not None:

            # SDK biasanya menyediakan as_image().
            try:

                result_image = part.as_image()

                if result_image is not None:
                    return result_image.convert("RGB")

            except Exception:
                pass

            # Fallback jika data tersedia sebagai bytes/base64.
            raw_data = getattr(
                inline_data,
                "data",
                None,
            )

            if raw_data:

                if isinstance(raw_data, bytes):

                    return Image.open(
                        io.BytesIO(raw_data)
                    ).convert("RGB")

                if isinstance(raw_data, str):

                    return Image.open(
                        io.BytesIO(
                            base64.b64decode(raw_data)
                        )
                    ).convert("RGB")

    # Fallback untuk SDK response yang menyediakan candidates.
    for candidate in getattr(response, "candidates", []) or []:

        content = getattr(
            candidate,
            "content",
            None,
        )

        for part in getattr(
            content,
            "parts",
            [],
        ) or []:

            inline_data = getattr(
                part,
                "inline_data",
                None,
            )

            if inline_data is None:
                continue

            raw_data = getattr(
                inline_data,
                "data",
                None,
            )

            if isinstance(raw_data, bytes):

                return Image.open(
                    io.BytesIO(raw_data)
                ).convert("RGB")

            if isinstance(raw_data, str):

                return Image.open(
                    io.BytesIO(
                        base64.b64decode(raw_data)
                    )
                ).convert("RGB")

    return None


# ============================================================
# GEMINI NATIVE IMAGE EDIT
# ============================================================

TIMESTAMP_REMOVE_PROMPT = """
Edit the provided photograph.

TASK:
Remove ONLY the GPS camera timestamp overlay from the photograph.

The timestamp is the camera-added overlay that may contain:
- GPS coordinates
- location/address
- date
- time
- timezone such as WIB/WITA/WIT
- multiple lines of timestamp text

IMPORTANT:
1. Do not crop the image.
2. Do not change the camera angle.
3. Do not change the composition.
4. Do not change any real objects in the photograph.
5. Do not change the documents, products, cables, remote, table,
   bed sheet, floor, walls, or any other natural photographic content.
6. Do not remove printed text that is part of a real physical object.
7. Remove ONLY the camera timestamp overlay.
8. Reconstruct the background behind the removed timestamp so it
   looks naturally photographed.
9. Preserve the original lighting, shadows, colors, texture,
   perspective, and photographic noise/grain.
10. Do not create new objects.
11. Do not add any text.
12. Do not leave blur, smudge, mosaic, clone artifacts, or visible
    evidence that something was removed.
13. Keep the result photorealistic.
14. Preserve the original image dimensions/aspect ratio as closely
    as the image editing model allows.

The camera timestamp may be near the bottom edge or another edge
of the photograph.

Again: REMOVE THE CAMERA TIMESTAMP ONLY.
Do not edit any other text printed on physical objects.
""".strip()


def remove_timestamp_with_gemini(
    image,
    api_key,
):
    """
    Kirim foto langsung ke Gemini image-editing model.
    Gemini melakukan detection + semantic removal + reconstruction.
    """

    client = genai.Client(
        api_key=api_key,
    )

    # Convert input ke PNG agar lossless saat dikirim ke model.
    image_buffer = io.BytesIO()

    image.save(
        image_buffer,
        format="PNG",
    )

    image_bytes = image_buffer.getvalue()

    # Legacy generate_content API dengan model image editing.
    # Google mendokumentasikan image editing sebagai
    # text + image -> image.
    response = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=[
            {
                "text": TIMESTAMP_REMOVE_PROMPT,
            },
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": image_bytes,
                }
            },
        ],
    )

    result_image = get_image_from_gemini_response(
        response
    )

    if result_image is None:

        response_text = getattr(
            response,
            "text",
            "",
        )

        raise RuntimeError(
            "Gemini tidak mengembalikan gambar hasil. "
            + (
                f"Response: {response_text[:500]}"
                if response_text
                else ""
            )
        )

    # Gemini image model dapat menghasilkan ukuran output
    # yang berbeda. Kita kembalikan ke dimensi asli agar
    # workflow dokumentasi tidak mengubah resolusi foto.
    if result_image.size != image.size:

        result_image = result_image.resize(
            image.size,
            Image.Resampling.LANCZOS,
        )

    return result_image


# ============================================================
# ZIP SAFETY + STRUCTURE
# ============================================================

def is_supported_image(filename):
    return (
        Path(filename).suffix.lower()
        in SUPPORTED_IMAGES
    )


def safe_zip_path(filename):
    """
    Mencegah path traversal seperti ../../file.
    """

    normalized = os.path.normpath(
        filename.replace("\\", "/")
    )

    if (
        normalized.startswith("../")
        or normalized == ".."
        or normalized.startswith("/")
        or os.path.isabs(normalized)
    ):
        return None

    return normalized


def get_zip_image_members(zip_bytes):
    """
    Mengambil daftar gambar dari ZIP.
    """

    with zipfile.ZipFile(
        io.BytesIO(zip_bytes),
        "r",
    ) as zf:

        members = []

        for info in zf.infolist():

            if info.is_dir():
                continue

            safe_name = safe_zip_path(
                info.filename
            )

            if safe_name is None:
                continue

            if is_supported_image(
                safe_name
            ):
                members.append(
                    info.filename
                )

        return members


def process_zip(
    zip_bytes,
    api_key,
    progress_callback=None,
):
    """
    Proses ZIP tanpa mengubah struktur.

    Input:
        Paket.zip
        ├── Folder A/
        │   ├── foto1.jpg
        │   └── foto2.jpg
        └── Folder B/
            └── foto3.jpg

    Output:
        Paket.zip
        ├── Folder A/
        │   ├── foto1.jpg
        │   └── foto2.jpg
        └── Folder B/
            └── foto3.jpg

    Nama file sama persis.
    Folder sama persis.
    """

    source = io.BytesIO(
        zip_bytes
    )

    output = io.BytesIO()

    with zipfile.ZipFile(
        source,
        "r",
    ) as input_zip:

        image_members = [
            info.filename
            for info in input_zip.infolist()
            if (
                not info.is_dir()
                and safe_zip_path(info.filename)
                and is_supported_image(info.filename)
            )
        ]

        if not image_members:

            raise ValueError(
                "ZIP tidak berisi JPG/JPEG/PNG/WEBP."
            )

        total = len(image_members)
        completed = 0

        with zipfile.ZipFile(
            output,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as output_zip:

            for info in input_zip.infolist():

                # Folder
                if info.is_dir():

                    safe_name = safe_zip_path(
                        info.filename
                    )

                    if safe_name is not None:

                        output_zip.writestr(
                            info,
                            b"",
                        )

                    continue

                safe_name = safe_zip_path(
                    info.filename
                )

                if safe_name is None:
                    continue

                original_bytes = input_zip.read(
                    info.filename
                )

                # --------------------------------------------
                # IMAGE
                # --------------------------------------------

                if info.filename in image_members:

                    try:

                        image = Image.open(
                            io.BytesIO(
                                original_bytes
                            )
                        )

                        image = ImageOps.exif_transpose(
                            image
                        ).convert("RGB")

                        result_image = remove_timestamp_with_gemini(
                            image,
                            api_key,
                        )

                        extension = Path(
                            info.filename
                        ).suffix.lower()

                        result_bytes = image_to_bytes(
                            result_image,
                            extension,
                        )

                        # Nama + path EXACT sama.
                        output_zip.writestr(
                            info.filename,
                            result_bytes,
                        )

                    except Exception:

                        # Jika Gemini gagal pada foto tertentu,
                        # foto asli tetap dimasukkan agar struktur
                        # ZIP tidak rusak/hilang.
                        output_zip.writestr(
                            info.filename,
                            original_bytes,
                        )

                    completed += 1

                    if progress_callback:

                        progress_callback(
                            completed,
                            total,
                        )

                # --------------------------------------------
                # NON-IMAGE FILE
                # --------------------------------------------

                else:

                    output_zip.writestr(
                        info.filename,
                        original_bytes,
                    )

    return (
        output.getvalue(),
        total,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-badge">GEMINI NATIVE IMAGE EDITING</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">🧹 Timestamp Remover</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-desc">'
    'Upload satu foto atau ZIP. Gemini langsung melakukan '
    '<b>semantic image editing</b> untuk menghapus timestamp '
    'dan merekonstruksi background secara natural.'
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
# INPUT
# ============================================================

st.markdown("### 1️⃣ Upload Foto atau ZIP")

uploaded = st.file_uploader(
    "Upload 1 foto atau 1 ZIP",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "zip",
    ],
    key=f"uploader_{st.session_state.uploader_version}",
)


input_mode = None
single_image = None
zip_bytes = None


if uploaded is not None:

    extension = Path(
        uploaded.name
    ).suffix.lower()

    # ========================================================
    # SINGLE PHOTO
    # ========================================================

    if extension != ".zip":

        input_mode = "single"

        try:

            single_image = load_pil_image(
                uploaded
            )

            width, height = (
                single_image.size
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📄 File",
                    uploaded.name,
                )

            with col2:
                st.metric(
                    "📐 Resolusi",
                    f"{width} × {height}",
                )

            with col3:
                st.metric(
                    "📦 Format",
                    extension.replace(".", "").upper(),
                )

            st.image(
                single_image,
                caption="Original",
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                f"❌ Foto tidak dapat dibaca: {error}"
            )

    # ========================================================
    # ZIP
    # ========================================================

    else:

        input_mode = "zip"

        zip_bytes = uploaded.getvalue()

        try:

            image_members = get_zip_image_members(
                zip_bytes
            )

            if not image_members:

                st.error(
                    "❌ ZIP tidak berisi foto yang didukung."
                )

            else:

                folders = sorted(
                    {
                        str(
                            Path(member).parent
                        )
                        for member in image_members
                        if str(
                            Path(member).parent
                        ) != "."
                    }
                )

                st.success(
                    f"📦 **{len(image_members)} foto** "
                    f"ditemukan dalam ZIP."
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "📷 Total Foto",
                        len(image_members),
                    )

                with col2:

                    st.metric(
                        "📁 Total Folder",
                        len(folders),
                    )

                with st.expander(
                    "📂 Lihat struktur ZIP",
                    expanded=False,
                ):

                    for member in image_members[:200]:

                        st.code(
                            member,
                            language=None,
                        )

                    if len(image_members) > 200:

                        st.caption(
                            f"... {len(image_members) - 200} foto lainnya."
                        )

        except Exception as error:

            st.error(
                f"❌ ZIP tidak dapat dibaca: {error}"
            )


# ============================================================
# REMOVE
# ============================================================

st.markdown("### 2️⃣ Remove Timestamp")

ready = (
    bool(gemini_api_key)
    and (
        (
            input_mode == "single"
            and single_image is not None
        )
        or
        (
            input_mode == "zip"
            and zip_bytes is not None
        )
    )
)


with st.container(border=True):

    st.markdown(
        """
        <div class="info-card">
        🧠 <b>Mode Gemini Native Editing</b><br>
        Foto dikirim langsung ke Gemini sebagai gambar untuk diedit.
        Gemini yang menentukan lokasi timestamp, menghapusnya,
        dan mengisi kembali background. Tidak ada rectangle mask
        atau threshold pixel dari OpenCV.
        </div>
        """,
        unsafe_allow_html=True,
    )

    remove_button = st.button(
        "🧹 REMOVE TIMESTAMP DENGAN GEMINI",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )


if remove_button:

    try:

        # ========================================================
        # SINGLE
        # ========================================================

        if input_mode == "single":

            with st.spinner(
                "🤖 Gemini sedang mengedit foto dan menghapus timestamp..."
            ):

                result_image = remove_timestamp_with_gemini(
                    single_image,
                    gemini_api_key,
                )

            output_name = uploaded.name

            output_bytes = image_to_bytes(
                result_image,
                Path(output_name).suffix,
            )

            st.session_state.remove_result = {
                "mode": "single",
                "image": result_image,
                "bytes": output_bytes,
                "name": output_name,
            }

            st.rerun()

        # ========================================================
        # ZIP
        # ========================================================

        else:

            progress = st.progress(
                0,
                text="📦 Menyiapkan ZIP...",
            )

            status = st.empty()

            def update_progress(
                current,
                total,
            ):

                ratio = (
                    current / total
                    if total
                    else 1
                )

                progress.progress(
                    ratio,
                    text=(
                        f"🧹 Gemini mengedit foto "
                        f"{current}/{total}"
                    ),
                )

                status.caption(
                    f"Foto {current} dari {total} sedang diproses."
                )

            with st.spinner(
                "🤖 Gemini sedang memproses seluruh foto..."
            ):

                result_zip, total = process_zip(
                    zip_bytes,
                    gemini_api_key,
                    update_progress,
                )

            progress.progress(
                1.0,
                text=f"✅ Selesai: {total} foto",
            )

            output_name = uploaded.name

            st.session_state.remove_result = {
                "mode": "zip",
                "bytes": result_zip,
                "name": output_name,
                "total": total,
            }

            st.rerun()

    except Exception as error:

        st.error(
            f"❌ Proses gagal: {error}"
        )


# ============================================================
# RESULT
# ============================================================

result = st.session_state.remove_result


if result is not None:

    st.markdown("### 3️⃣ Hasil")

    with st.container(border=True):

        if result["mode"] == "single":

            st.success(
                "🎉 Timestamp berhasil diproses oleh Gemini."
            )

            st.image(
                result["image"],
                caption="Hasil Gemini",
                use_container_width=True,
            )

            col1, col2 = st.columns([2, 1])

            with col1:

                st.download_button(
                    "⬇️ DOWNLOAD FOTO",
                    data=result["bytes"],
                    file_name=result["name"],
                    mime=(
                        "image/jpeg"
                        if Path(
                            result["name"]
                        ).suffix.lower()
                        in [".jpg", ".jpeg"]
                        else "image/png"
                    ),
                    type="primary",
                    use_container_width=True,
                )

            with col2:

                if st.button(
                    "🔄 Foto Baru",
                    use_container_width=True,
                ):

                    reset_app()
                    st.rerun()

        else:

            st.success(
                f"🎉 **ZIP selesai diproses.** "
                f"{result['total']} foto diproses."
            )

            st.markdown(
                """
                <div class="info-card">
                📁 Struktur folder dan subfolder tetap sama.<br>
                📄 Nama setiap foto tetap sama persis seperti input.<br>
                📦 ZIP output menggunakan nama ZIP input.
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns([2, 1])

            with col1:

                st.download_button(
                    "⬇️ DOWNLOAD ZIP",
                    data=result["bytes"],
                    file_name=result["name"],
                    mime="application/zip",
                    type="primary",
                    use_container_width=True,
                )

            with col2:

                if st.button(
                    "🔄 ZIP Baru",
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
        <b>Timestamp Remover V9.0</b><br>
        Gemini Native Image Editing
    </div>
    """,
    unsafe_allow_html=True,
)
