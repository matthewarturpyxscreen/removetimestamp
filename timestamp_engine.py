import io
import re
import traceback
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any

import cv2
import numpy as np
from PIL import Image

# =========================
# SETTINGS (ported from V2.9)
# =========================
BOTTOM_OCR_START = 0.50
OCR_MIN_CONFIDENCE = 0.08
MASK_PADDING_X = 3
MASK_PADDING_Y = 3
MASK_DILATION = 3
MAX_DIMENSION = 2400

OCR_PASS1_TEXT_THRESHOLD = 0.18
OCR_PASS1_LOW_TEXT = 0.08
OCR_PASS1_LINK_THRESHOLD = 0.15

OCR_PASS2_TEXT_THRESHOLD = 0.10
OCR_PASS2_LOW_TEXT = 0.03
OCR_PASS2_LINK_THRESHOLD = 0.10

TEXT_EDGE_LOW = 40
TEXT_EDGE_HIGH = 120
MIN_TEXT_PIXEL_RATIO = 0.03

LINE_RESTORE_CONTEXT_MARGIN = 120
LINE_RESTORE_HOUGH_THRESHOLD = 60
LINE_RESTORE_MIN_LENGTH = 80
LINE_RESTORE_MAX_GAP = 15
LINE_RESTORE_THICKNESS = 2

SECONDARY_WATERMARK_COLORS = [
    {
        "name": "biru",
        "lower_hsv": [85, 60, 120],
        "upper_hsv": [135, 255, 255],
    },
]

MONTH_NAMES = (
    r"(?:Jan|Feb|Mar|Apr|Mei|May|Jun|Jul|Agu|Aug|Ags|"
    r"Sep|Okt|Oct|Nov|Des|Dec|"
    r"Januari|Februari|Maret|April|Juni|Juli|Agustus|"
    r"September|Oktober|November|Desember)"
)

Logger = Optional[Callable[[str], None]]


def _log(logger: Logger, message: str) -> None:
    if logger:
        logger(message)


def normalize_ocr_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    replacements = {
        "，": ",", "。": ".", "；": ";", "：": ":",
        "–": "-", "—": "-", "−": "-",
        "WlB": "WIB", "WIB.": "WIB", "wib": "WIB",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_coordinate_text(text):
    text = normalize_ocr_text(text)

    match = re.search(r"([+-]?\d{1,3}\.\d+)\s*,\s*([+-]?\d{1,3}\.\d+)", text)
    if match:
        return match.group(1) + ", " + match.group(2)

    match = re.search(
        r"([+-]?\d{1,3})\s*[,\.]\s*(\d{3,10})\s*[,\.]\s*([+-]?\d{1,3}\.\d+)",
        text,
    )
    if match:
        return match.group(1) + "." + match.group(2) + ", " + match.group(3)

    match = re.search(
        r"([+-]?\d{1,3})\s+(\d{3,10})\s*,\s*([+-]?\d{1,3}\.\d+)", text
    )
    if match:
        return match.group(1) + "." + match.group(2) + ", " + match.group(3)

    return None


def extract_coordinate(text):
    normalized = normalize_coordinate_text(text)
    if normalized is None:
        return None

    match = re.match(r"^([+-]?\d{1,3}\.\d+)\s*,\s*([+-]?\d{1,3}\.\d+)$", normalized)
    if not match:
        return None

    try:
        lat = float(match.group(1))
        lon = float(match.group(2))
    except (TypeError, ValueError):
        return None

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None
    return normalized


def _normalize_datetime_separators(text):
    text = normalize_ocr_text(text)
    text = re.sub(r"/{2,}", "/", text)
    text = re.sub(r"(\d{1,2})\.(\d{2}):(\d{2})", r"\1:\2:\3", text)
    text = re.sub(r"(\d{1,2}):(\d{2})\.(\d{2})", r"\1:\2:\3", text)
    text = re.sub(r"(\d{1,2})\.(\d{2})\.(\d{2})", r"\1:\2:\3", text)
    return text


def datetime_match(text):
    cleaned = _normalize_datetime_separators(text)
    pattern = re.compile(
        r"\b\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4}\s+"
        r"\d{1,2}\s*:\s*\d{2}\s*:\s*\d{2}\s*(?:WIB|WlB)\b",
        re.IGNORECASE,
    )
    return pattern.search(cleaned)


def datetime_match_ampm(text):
    cleaned = normalize_ocr_text(text)
    pattern = re.compile(
        r"\b\d{1,2}\s+" + MONTH_NAMES + r"\s+\d{4}\s+"
        r"\d{1,2}\s*[.:]\s*\d{2}\s*[.:]\s*\d{2}\s*(?:AM|PM)\b",
        re.IGNORECASE,
    )
    return pattern.search(cleaned)


def datetime_match_any(text):
    return datetime_match(text) or datetime_match_ampm(text)


def group_ocr_lines(results):
    items = []
    for bbox, text, confidence in results:
        if confidence < OCR_MIN_CONFIDENCE:
            continue
        pts = np.asarray(bbox, dtype=np.float32)
        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))
        items.append({
            "bbox": pts,
            "text": normalize_ocr_text(text),
            "confidence": float(confidence),
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "center_y": (y_min + y_max) / 2,
            "height": y_max - y_min,
        })

    if not items:
        return []

    items.sort(key=lambda item: item["center_y"])
    lines = []
    for item in items:
        assigned = False
        for line in lines:
            avg_y = np.mean([x["center_y"] for x in line])
            avg_height = np.mean([x["height"] for x in line])
            tolerance = max(10, avg_height * 0.65)
            if abs(item["center_y"] - avg_y) <= tolerance:
                line.append(item)
                assigned = True
                break
        if not assigned:
            lines.append([item])

    for line in lines:
        line.sort(key=lambda item: item["x_min"])
    return lines


def enhance_contrast(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _mask_from_hsv_range(image_bgr, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))


def _split_mask_into_lines(mask, min_line_height=10, min_gap=8):
    row_sum = (mask > 0).sum(axis=1)
    has_text = row_sum > 2
    segments = []
    in_seg = False
    start = 0

    for y, value in enumerate(has_text):
        if value and not in_seg:
            start = y
            in_seg = True
        elif not value and in_seg:
            segments.append((start, y))
            in_seg = False
    if in_seg:
        segments.append((start, len(has_text)))

    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    return [s for s in merged if (s[1] - s[0]) >= min_line_height]


def isolate_and_split_lines(image_bgr, lower_hsv=None, upper_hsv=None):
    lower_hsv = lower_hsv or [0, 0, 225]
    upper_hsv = upper_hsv or [180, 25, 255]
    mask = _mask_from_hsv_range(image_bgr, lower_hsv, upper_hsv)
    segments = _split_mask_into_lines(mask, min_line_height=15)

    lines_out = []
    pad = 10
    img_h, img_w = mask.shape[:2]
    for y1, y2 in segments:
        yy1 = max(0, y1 - pad)
        yy2 = min(img_h, y2 + pad)
        line_canvas = np.full((yy2 - yy1, img_w, 3), 255, dtype=np.uint8)
        line_mask = mask[yy1:yy2, :]
        line_canvas[line_mask > 0] = (0, 0, 0)
        lines_out.append((yy1, yy2, line_canvas))
    return lines_out


def _quick_color_present(image_bgr, lower_hsv, upper_hsv, min_pct=0.15):
    mask = _mask_from_hsv_range(image_bgr, lower_hsv, upper_hsv)
    return (mask > 0).mean() * 100 >= min_pct


def detect_secondary_watermark(resized, reader, crop_start, inv_scale, width, height, logger=None):
    for color in SECONDARY_WATERMARK_COLORS:
        if not _quick_color_present(resized, color["lower_hsv"], color["upper_hsv"]):
            continue

        line_segments = isolate_and_split_lines(resized, color["lower_hsv"], color["upper_hsv"])
        if not line_segments:
            continue

        found_datetime = False
        all_boxes = []
        for yy1, yy2, line_canvas in line_segments:
            rgb_line = cv2.cvtColor(line_canvas, cv2.COLOR_BGR2RGB)
            raw = reader.readtext(
                rgb_line,
                paragraph=True,
                text_threshold=OCR_PASS1_TEXT_THRESHOLD,
                low_text=OCR_PASS1_LOW_TEXT,
                link_threshold=OCR_PASS1_LINK_THRESHOLD,
            )
            line_has_content = False
            for item in raw:
                if len(item) == 2:
                    bbox, text = item
                else:
                    bbox, text, _confidence = item
                line_has_content = True
                if datetime_match_any(text) is not None:
                    found_datetime = True

            if line_has_content:
                line_mask_gray = cv2.cvtColor(line_canvas, cv2.COLOR_BGR2GRAY)
                nonwhite = np.where(line_mask_gray < 250)
                if len(nonwhite[1]) > 0:
                    all_boxes.append((int(nonwhite[1].min()), yy1, int(nonwhite[1].max()), yy2))

        if found_datetime and all_boxes:
            _log(logger, f"Watermark sekunder ({color['name']}) terdeteksi")
            mask = np.zeros((height, width), dtype=np.uint8)
            for x_min, y_min, x_max, y_max in all_boxes:
                ox_min = max(0, int(x_min * inv_scale) - MASK_PADDING_X)
                ox_max = min(width - 1, int(x_max * inv_scale) + MASK_PADDING_X)
                oy_min = max(0, int(y_min * inv_scale) + crop_start - MASK_PADDING_Y)
                oy_max = min(height - 1, int(y_max * inv_scale) + crop_start + MASK_PADDING_Y)
                cv2.rectangle(mask, (ox_min, oy_min), (ox_max, oy_max), 255, -1)
            return mask
    return None


def _precise_text_mask_in_box(image_bgr, x_min, y_min, x_max, y_max):
    crop = image_bgr[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, TEXT_EDGE_LOW, TEXT_EDGE_HIGH)
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_or(edges, bright)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=1)

    if (combined > 0).mean() < MIN_TEXT_PIXEL_RATIO:
        return np.full(gray.shape, 255, dtype=np.uint8)
    return combined


def build_precise_mask(image_bgr, timestamp_lines, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in timestamp_lines:
        for item in line:
            pts = np.asarray(item["bbox"], dtype=np.float32)
            x_min = max(0, int(np.floor(np.min(pts[:, 0]))) - MASK_PADDING_X)
            y_min = max(0, int(np.floor(np.min(pts[:, 1]))) - MASK_PADDING_Y)
            x_max = min(width - 1, int(np.ceil(np.max(pts[:, 0]))) + MASK_PADDING_X)
            y_max = min(height - 1, int(np.ceil(np.max(pts[:, 1]))) + MASK_PADDING_Y)
            if x_max <= x_min or y_max <= y_min:
                continue
            precise = _precise_text_mask_in_box(image_bgr, x_min, y_min, x_max, y_max)
            if precise is None:
                continue
            mask[y_min:y_max, x_min:x_max] = cv2.bitwise_or(
                mask[y_min:y_max, x_min:x_max], precise
            )

    if MASK_DILATION > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATION, MASK_DILATION))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def detect_timestamp(image_bgr, reader, logger=None):
    height, width = image_bgr.shape[:2]
    crop_start = int(height * BOTTOM_OCR_START)
    crop = image_bgr[crop_start:height, 0:width]
    crop_h, crop_w = crop.shape[:2]
    scale = min(1.0, MAX_DIMENSION / max(crop_h, crop_w))
    resized = (
        cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0 else crop
    )
    inv_scale = 1.0 / scale

    def run_pass(image_for_ocr, text_threshold, low_text, link_threshold):
        rgb = cv2.cvtColor(image_for_ocr, cv2.COLOR_BGR2RGB)
        raw_results = reader.readtext(
            rgb,
            paragraph=False,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            mag_ratio=1.0,
        )
        adjusted = []
        for bbox, text, confidence in raw_results:
            pts = np.asarray(bbox, dtype=np.float32)
            pts *= inv_scale
            pts[:, 1] += crop_start
            adjusted.append((pts.tolist(), text, confidence))
        return adjusted

    def anchors_found(lines):
        if not lines:
            return False
        has_coord = any(extract_coordinate(" ".join(i["text"] for i in line)) is not None for line in lines)
        has_dt = any(datetime_match_any(" ".join(i["text"] for i in line)) is not None for line in lines)
        return has_coord and has_dt

    adjusted = run_pass(resized, OCR_PASS1_TEXT_THRESHOLD, OCR_PASS1_LOW_TEXT, OCR_PASS1_LINK_THRESHOLD)
    lines = group_ocr_lines(adjusted) if adjusted else []
    used_fallback = None

    if not anchors_found(lines):
        _log(logger, "Pass 1 OCR belum lengkap → fallback pass 2")
        enhanced = enhance_contrast(resized)
        adjusted_fb = run_pass(enhanced, OCR_PASS2_TEXT_THRESHOLD, OCR_PASS2_LOW_TEXT, OCR_PASS2_LINK_THRESHOLD)
        lines_fb = group_ocr_lines(adjusted_fb) if adjusted_fb else []
        if anchors_found(lines_fb) or len(lines_fb) > len(lines):
            lines = lines_fb
            used_fallback = "contrast"

    if not anchors_found(lines):
        _log(logger, "Pass 2 belum lengkap → fallback pass 3 (isolasi teks putih)")
        adjusted_pass3 = []
        for yy1, yy2, line_canvas in isolate_and_split_lines(resized):
            rgb_line = cv2.cvtColor(line_canvas, cv2.COLOR_BGR2RGB)
            raw = reader.readtext(
                rgb_line,
                paragraph=True,
                text_threshold=OCR_PASS1_TEXT_THRESHOLD,
                low_text=OCR_PASS1_LOW_TEXT,
                link_threshold=OCR_PASS1_LINK_THRESHOLD,
            )
            for item in raw:
                if len(item) == 2:
                    bbox, text = item
                    confidence = 0.99
                else:
                    bbox, text, confidence = item
                pts = np.asarray(bbox, dtype=np.float32)
                pts[:, 1] += yy1
                adjusted_pass3.append((pts.tolist(), text, confidence))

        adjusted_original = []
        for bbox, text, confidence in adjusted_pass3:
            pts = np.asarray(bbox, dtype=np.float32)
            pts *= inv_scale
            pts[:, 1] += crop_start
            adjusted_original.append((pts.tolist(), text, confidence))

        lines_p3 = group_ocr_lines(adjusted_original) if adjusted_original else []
        if anchors_found(lines_p3) or len(lines_p3) > len(lines):
            lines = lines_p3
            used_fallback = "white_isolation"

    if not lines:
        return None

    for idx, line in enumerate(lines, 1):
        _log(logger, f"OCR [{idx}] " + " ".join(item["text"] for item in line))

    coordinate_index = None
    coordinate_value = None
    for i, line in enumerate(lines):
        coordinate = extract_coordinate(" ".join(item["text"] for item in line))
        if coordinate is not None:
            coordinate_index = i
            coordinate_value = coordinate
            break

    datetime_index = None
    datetime_value = None
    for i, line in enumerate(lines):
        if coordinate_index is not None and i <= coordinate_index:
            continue
        match = datetime_match_any(" ".join(item["text"] for item in line))
        if match:
            datetime_index = i
            datetime_value = match.group(0)
            break

    if coordinate_index is None or datetime_index is None or datetime_index < coordinate_index:
        return None

    timestamp_lines = lines[coordinate_index:datetime_index + 1]
    mask = build_precise_mask(image_bgr, timestamp_lines, width, height)

    secondary_mask = detect_secondary_watermark(
        resized, reader, crop_start, inv_scale, width, height, logger=logger
    )
    if secondary_mask is not None:
        mask = cv2.bitwise_or(mask, secondary_mask)

    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    _log(logger, f"Mask: x={x}, y={y}, w={w}, h={h}")

    return {
        "mask": mask,
        "coordinate": coordinate_value,
        "datetime": datetime_value,
        "lines": timestamp_lines,
        "fallback": used_fallback,
    }


def run_simple_lama(image: Image.Image, mask: np.ndarray, lama, logger=None):
    points = cv2.findNonZero(mask)
    if points is None:
        return image.copy()

    x, y, w, h = cv2.boundingRect(points)
    padding_x = max(80, int(w * 0.15))
    padding_y = max(80, int(h * 0.50))
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(image.width, x + w + padding_x)
    y2 = min(image.height, y + h + padding_y)

    crop_image = image.crop((x1, y1, x2, y2))
    crop_mask = Image.fromarray(mask[y1:y2, x1:x2]).convert("L")
    _log(logger, "Running SimpleLama...")
    result_crop = lama(crop_image, crop_mask)
    result = image.copy()
    result.paste(result_crop, (x1, y1))
    return result


def _get_line_color_sample(image_bgr, x1, y1, x2, y2, sample_width=3):
    h, w = image_bgr.shape[:2]
    num_samples = max(int(np.hypot(x2 - x1, y2 - y1)), 1)
    colors = []
    for t in np.linspace(0, 1, num_samples):
        sx = int(x1 + (x2 - x1) * t)
        sy = int(y1 + (y2 - y1) * t)
        x0, xs1 = max(0, sx - sample_width), min(w, sx + sample_width + 1)
        y0, ys1 = max(0, sy - sample_width), min(h, sy + sample_width + 1)
        patch = image_bgr[y0:ys1, x0:xs1]
        if patch.size > 0:
            colors.append(patch.reshape(-1, 3).mean(axis=0))
    return np.mean(colors, axis=0) if colors else None


def restore_straight_lines(
    original_bgr,
    result_bgr,
    mask,
    context_margin=LINE_RESTORE_CONTEXT_MARGIN,
    hough_threshold=LINE_RESTORE_HOUGH_THRESHOLD,
    min_line_length=LINE_RESTORE_MIN_LENGTH,
    max_line_gap=LINE_RESTORE_MAX_GAP,
    line_thickness=LINE_RESTORE_THICKNESS,
    logger=None,
):
    """Restores straight lines across the inpainted mask.

    V2.9.1 FIX:
    - Robust to HoughLinesP output shaped as (N,1,4) or (N,4).
    - Uses correct cv2.clipLine rect: (x, y, width, height).
    - Any malformed Hough line is skipped instead of crashing the image.
    """
    height, width = mask.shape[:2]
    points = cv2.findNonZero(mask)
    if points is None:
        return result_bgr.copy()

    x, y, w, h = [int(v) for v in cv2.boundingRect(points)]
    cx1 = max(0, x - context_margin)
    cy1 = max(0, y - context_margin)
    cx2 = min(width, x + w + context_margin)
    cy2 = min(height, y + h + context_margin)

    context_crop = original_bgr[cy1:cy2, cx1:cx2]
    context_mask = mask[cy1:cy2, cx1:cx2]
    gray = cv2.cvtColor(context_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges[context_mask > 0] = 0

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    output = result_bgr.copy()
    if lines is None:
        _log(logger, "Restorasi garis: tidak ada garis yang perlu dipulihkan")
        return output

    # Correct OpenCV rect format: x, y, width, height.
    mask_rect = (x, y, w, h)
    restored_count = 0

    # FIX: normalize all Hough outputs to rows of 4 coordinates.
    lines_flat = np.asarray(lines).reshape(-1, 4)

    for coords in lines_flat:
        if np.asarray(coords).size != 4:
            continue
        lx1, ly1, lx2, ly2 = map(int, np.asarray(coords).reshape(-1)[:4])
        gx1, gy1 = lx1 + cx1, ly1 + cy1
        gx2, gy2 = lx2 + cx1, ly2 + cy1

        dx, dy = gx2 - gx1, gy2 - gy1
        length = float(np.hypot(dx, dy))
        if length < 1:
            continue

        ext = max(width, height)
        ux, uy = dx / length, dy / length
        ex1, ey1 = int(gx1 - ux * ext), int(gy1 - uy * ext)
        ex2, ey2 = int(gx2 + ux * ext), int(gy2 + uy * ext)

        try:
            clipped = cv2.clipLine(mask_rect, (ex1, ey1), (ex2, ey2))
        except cv2.error:
            continue

        if not clipped or not bool(clipped[0]):
            continue

        # Defensive parsing: OpenCV returns (retval, pt1, pt2), but keep it safe.
        if len(clipped) < 3:
            continue
        pt_a = np.asarray(clipped[1]).reshape(-1)
        pt_b = np.asarray(clipped[2]).reshape(-1)
        if pt_a.size < 2 or pt_b.size < 2:
            continue
        cx_a, cy_a = map(int, pt_a[:2])
        cx_b, cy_b = map(int, pt_b[:2])

        color = _get_line_color_sample(original_bgr, gx1, gy1, gx2, gy2)
        if color is None:
            continue

        pad = 15
        px1, py1 = int(cx_a - ux * pad), int(cy_a - uy * pad)
        px2, py2 = int(cx_b + ux * pad), int(cy_b + uy * pad)
        cv2.line(
            output,
            (px1, py1),
            (px2, py2),
            tuple(int(np.clip(c, 0, 255)) for c in color),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
        restored_count += 1

    if restored_count:
        blurred = cv2.GaussianBlur(output, (3, 3), 0)
        dil = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        output = np.where(dil[..., None] > 0, blurred, output)

    _log(logger, f"Restorasi garis selesai: {restored_count} garis")
    return output


def safe_restore_straight_lines(original_bgr, result_bgr, mask, logger=None):
    """Fail-safe wrapper: restoration is optional and must never invalidate LaMa output."""
    try:
        return restore_straight_lines(original_bgr, result_bgr, mask, logger=logger), None
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        _log(logger, f"Restorasi garis dilewati karena error: {detail}")
        return result_bgr.copy(), detail


def image_to_bytes(image: Image.Image, filename: str) -> tuple[bytes, str]:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "png"
    buf = io.BytesIO()
    if ext in ("jpg", "jpeg"):
        image.save(buf, format="JPEG", quality=95, subsampling=0)
        mime = "image/jpeg"
    elif ext == "webp":
        image.save(buf, format="WEBP", quality=95)
        mime = "image/webp"
    elif ext == "bmp":
        image.save(buf, format="BMP")
        mime = "image/bmp"
    else:
        image.save(buf, format="PNG")
        mime = "image/png"
    return buf.getvalue(), mime


@dataclass
class ProcessResult:
    ok: bool
    status: str
    filename: str
    original: Optional[Image.Image] = None
    result: Optional[Image.Image] = None
    mask: Optional[np.ndarray] = None
    coordinate: Optional[str] = None
    datetime: Optional[str] = None
    fallback: Optional[str] = None
    restore_warning: Optional[str] = None
    logs: Optional[List[str]] = None
    error: Optional[str] = None


def process_image_bytes(data: bytes, filename: str, reader, lama, restore_lines=True) -> ProcessResult:
    logs: List[str] = []
    logger = logs.append
    try:
        original = Image.open(io.BytesIO(data)).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
        detection = detect_timestamp(image_bgr, reader, logger=logger)

        if detection is None:
            return ProcessResult(
                ok=False,
                status="skip",
                filename=filename,
                original=original,
                logs=logs,
                error="Timestamp block tidak ditemukan.",
            )

        mask = detection["mask"]
        result_image = run_simple_lama(original, mask, lama, logger=logger)
        restore_warning = None

        if restore_lines:
            result_bgr = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            result_bgr, restore_warning = safe_restore_straight_lines(
                image_bgr, result_bgr, mask, logger=logger
            )
            result_image = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

        return ProcessResult(
            ok=True,
            status="success",
            filename=filename,
            original=original,
            result=result_image,
            mask=mask,
            coordinate=detection["coordinate"],
            datetime=detection["datetime"],
            fallback=detection.get("fallback"),
            restore_warning=restore_warning,
            logs=logs,
        )

    except Exception as exc:
        return ProcessResult(
            ok=False,
            status="error",
            filename=filename,
            logs=logs,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
