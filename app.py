import io
import zipfile

import easyocr
import streamlit as st
import torch
from simple_lama_inpainting import SimpleLama

from timestamp_engine import process_image_bytes, image_to_bytes

st.set_page_config(
    page_title="Timestamp Remover V2.9.1",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 Timestamp Remover V2.9.1")
st.caption("EasyOCR + SimpleLama • Streamlit • HoughLinesP fix + safe line restoration")


@st.cache_resource(show_spinner=False)
def load_models():
    gpu_available = torch.cuda.is_available()
    reader = easyocr.Reader(["en"], gpu=gpu_available, verbose=False)
    lama = SimpleLama()
    return reader, lama, gpu_available


with st.spinner("Memuat EasyOCR dan SimpleLama..."):
    reader, lama, gpu_available = load_models()

if gpu_available:
    st.success(f"GPU terdeteksi: {torch.cuda.get_device_name(0)}")
else:
    st.info("GPU tidak tersedia. Aplikasi tetap berjalan menggunakan CPU.")

with st.sidebar:
    st.header("Pengaturan")
    restore_lines = st.checkbox(
        "Restorasi garis lurus",
        value=True,
        help="Mencoba menyambungkan nat lantai/pola garis setelah inpainting. Jika fungsi ini error, hasil LaMa tetap disimpan.",
    )
    show_logs = st.checkbox("Tampilkan log OCR", value=False)
    show_mask = st.checkbox("Tampilkan mask", value=False)

uploaded_files = st.file_uploader(
    "Upload satu atau banyak foto",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True,
)

if "results" not in st.session_state:
    st.session_state.results = []

if uploaded_files:
    st.write(f"**{len(uploaded_files)} foto siap diproses.**")

    if st.button("🚀 Proses Foto", type="primary", use_container_width=True):
        st.session_state.results = []
        progress = st.progress(0, text="Menyiapkan proses...")

        for idx, uploaded in enumerate(uploaded_files, start=1):
            progress.progress(
                (idx - 1) / len(uploaded_files),
                text=f"Memproses {idx}/{len(uploaded_files)}: {uploaded.name}",
            )
            result = process_image_bytes(
                uploaded.getvalue(),
                uploaded.name,
                reader,
                lama,
                restore_lines=restore_lines,
            )
            st.session_state.results.append(result)

        progress.progress(1.0, text="Selesai")

results = st.session_state.results

if results:
    success = sum(r.status == "success" for r in results)
    skipped = sum(r.status == "skip" for r in results)
    errors = sum(r.status == "error" for r in results)

    c1, c2, c3 = st.columns(3)
    c1.metric("Berhasil", success)
    c2.metric("Skip", skipped)
    c3.metric("Error", errors)

    zip_buffer = io.BytesIO()
    has_zip_items = False

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for result_idx, result in enumerate(results):
            with st.container(border=True):
                st.subheader(result.filename)

                if result.status == "success":
                    st.success("Timestamp berhasil diproses")
                    info1, info2, info3 = st.columns(3)
                    info1.write(f"**Coordinate:** {result.coordinate or '-'}")
                    info2.write(f"**Datetime:** {result.datetime or '-'}")
                    info3.write(f"**OCR:** {result.fallback or 'pass 1'}")

                    if result.restore_warning:
                        st.warning(
                            "Inpainting berhasil, tetapi restorasi garis dilewati: "
                            + result.restore_warning
                        )

                    before_col, after_col = st.columns(2)
                    with before_col:
                        st.markdown("**BEFORE**")
                        st.image(result.original, use_container_width=True)
                    with after_col:
                        st.markdown("**AFTER REMOVE**")
                        st.image(result.result, use_container_width=True)

                    if show_mask and result.mask is not None:
                        st.markdown("**MASK**")
                        st.image(result.mask, clamp=True, use_container_width=True)

                    output_bytes, mime = image_to_bytes(result.result, result.filename)
                    st.download_button(
                        "⬇️ Download hasil",
                        data=output_bytes,
                        file_name=result.filename,
                        mime=mime,
                        key=f"download_{result_idx}_{result.filename}",
                        use_container_width=True,
                    )
                    zf.writestr(result.filename, output_bytes)
                    has_zip_items = True

                elif result.status == "skip":
                    st.warning(result.error or "Timestamp tidak ditemukan.")
                    if result.original is not None:
                        st.image(result.original, use_container_width=True)

                else:
                    st.error("Proses gagal")
                    st.code(result.error or "Unknown error")

                if show_logs and result.logs:
                    with st.expander("Log OCR / processing"):
                        st.code("\n".join(result.logs))

    if has_zip_items:
        st.download_button(
            "📦 Download Semua Hasil (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="timestamp_removed_v291.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )
else:
    st.info("Upload foto lalu klik **Proses Foto**.")

st.divider()
st.caption(
    "V2.9.1: memperbaiki parsing output cv2.HoughLinesP dan menambahkan fail-safe "
    "agar error restorasi garis tidak menggagalkan hasil inpainting."
)
