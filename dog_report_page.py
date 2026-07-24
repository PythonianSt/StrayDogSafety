import base64
import csv
import io
import json
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import pydeck as pdk
import requests
import streamlit as st
from PIL import Image, ImageOps
from openai import OpenAI
from streamlit_geolocation import streamlit_geolocation


# =========================================================
# CONSTANTS
# =========================================================

BKK_TZ = ZoneInfo("Asia/Bangkok")

CSV_FIELDS = [
    "report_id",
    "timestamp_bkk",
    "date_bkk",
    "time_bkk",
    "latitude",
    "longitude",
    "location_accuracy_m",
    "image_count",
    "risk_color",
    "risk_level_th",
    "risk_score",
    "dog_count",
    "dog_count_description",
    "observed_behavior",
    "visible_risk_factors",
    "recommendation",
    "confidence",
    "image_quality",
    "needs_human_review",
    "model",
]

RISK_DISPLAY = {
    "green": {
        "thai": "สีเขียว",
        "icon": "🟢",
        "title": "ความเสี่ยงต่ำ",
        "background": "#E8F5E9",
        "border": "#2E7D32",
    },
    "yellow": {
        "thai": "สีเหลือง",
        "icon": "🟡",
        "title": "ควรระมัดระวัง",
        "background": "#FFF8E1",
        "border": "#F9A825",
    },
    "red": {
        "thai": "สีแดง",
        "icon": "🔴",
        "title": "ความเสี่ยงสูง",
        "background": "#FFEBEE",
        "border": "#C62828",
    },
}


# =========================================================
# SECRETS
# =========================================================

def get_secret(name: str, default=None):
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        return default


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4.1-mini")

GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
GITHUB_OWNER = get_secret("GITHUB_OWNER")
GITHUB_REPO = get_secret("GITHUB_REPO")
GITHUB_BRANCH = get_secret("GITHUB_BRANCH", "main")
GITHUB_CSV_PATH = get_secret(
    "GITHUB_CSV_PATH",
    "data/stray_dog_reports.csv",
)


# =========================================================
# SESSION STATE
# =========================================================

if "latitude" not in st.session_state:
    st.session_state.latitude = None

if "longitude" not in st.session_state:
    st.session_state.longitude = None

if "location_accuracy" not in st.session_state:
    st.session_state.location_accuracy = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "saved_report_id" not in st.session_state:
    st.session_state.saved_report_id = None


# =========================================================
# IMAGE PROCESSING
# =========================================================

def prepare_image(
    uploaded_file,
    max_dimension: int = 1600,
    jpeg_quality: int = 85,
) -> dict:
    """
    เปิดภาพ ปรับ orientation ลบ metadata ย่อภาพ และแปลงเป็น JPEG

    ภาพที่ผ่านฟังก์ชันนี้ใช้แสดงและส่งให้ OpenAI
    โดยไม่มี EXIF เดิมติดไปด้วย
    """

    raw_bytes = uploaded_file.getvalue()

    if not raw_bytes:
        raise ValueError("ไฟล์ภาพว่าง")

    image = Image.open(io.BytesIO(raw_bytes))
    image = ImageOps.exif_transpose(image)

    if image.mode not in ("RGB", "L"):
        background = Image.new("RGB", image.size, "white")

        if "A" in image.getbands():
            background.paste(image, mask=image.getchannel("A"))
        else:
            background.paste(image)

        image = background

    elif image.mode == "L":
        image = image.convert("RGB")

    image.thumbnail(
        (max_dimension, max_dimension),
        Image.Resampling.LANCZOS,
    )

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
    )

    processed_bytes = output.getvalue()

    return {
        "bytes": processed_bytes,
        "mime_type": "image/jpeg",
        "width": image.width,
        "height": image.height,
    }


def bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


# =========================================================
# JSON PARSING
# =========================================================

def extract_json_object(text: str) -> dict:
    """
    รองรับทั้ง JSON ตรงๆ และ JSON ที่ถูกครอบด้วย Markdown code fence
    """

    if not text:
        raise ValueError("GPT ไม่ได้ส่งผลวิเคราะห์กลับมา")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1)
        cleaned = cleaned.replace("```JSON", "", 1)
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("ไม่พบ JSON ที่สมบูรณ์ในคำตอบของ GPT")

        return json.loads(cleaned[start:end + 1])


def normalize_analysis(result: dict) -> dict:
    risk_color = str(result.get("risk_color", "")).strip().lower()

    if risk_color not in {"green", "yellow", "red"}:
        risk_color = "yellow"

    try:
        risk_score = int(result.get("risk_score", 2))
    except (TypeError, ValueError):
        risk_score = 2

    risk_score = max(1, min(risk_score, 3))

    try:
        dog_count = int(result.get("dog_count", 0))
    except (TypeError, ValueError):
        dog_count = 0

    confidence = str(result.get("confidence", "low")).strip().lower()

    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

    image_quality = str(
        result.get("image_quality", "unclear")
    ).strip().lower()

    if image_quality not in {"clear", "partly_clear", "unclear"}:
        image_quality = "unclear"

    return {
        "risk_color": risk_color,
        "risk_level_th": RISK_DISPLAY[risk_color]["thai"],
        "risk_score": risk_score,
        "dog_count": max(0, dog_count),
        "dog_count_description": str(
            result.get("dog_count_description", "")
        ).strip(),
        "observed_behavior": str(
            result.get("observed_behavior", "")
        ).strip(),
        "visible_risk_factors": str(
            result.get("visible_risk_factors", "")
        ).strip(),
        "recommendation": str(
            result.get("recommendation", "")
        ).strip(),
        "confidence": confidence,
        "image_quality": image_quality,
        "needs_human_review": bool(
            result.get("needs_human_review", False)
        ),
    }


# =========================================================
# GPT VISION ANALYSIS
# =========================================================

def analyze_dog_images(processed_images: list[dict]) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "ยังไม่ได้ตั้งค่า OPENAI_API_KEY ใน Streamlit Secrets"
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    assessment_prompt = """
คุณเป็นระบบช่วยประเมินความเสี่ยงจากสุนัขในพื้นที่มหาวิทยาลัย
วิเคราะห์เฉพาะสิ่งที่มองเห็นได้จากภาพทั้งหมดร่วมกัน

เป้าหมายคือจัดระดับความเสี่ยงต่อผู้ใช้ทางเดินเป็นหนึ่งสีเท่านั้น:

GREEN:
- สุนัขอยู่สงบ นอน นั่ง หรือเดินตามปกติ
- อยู่ห่างจากผู้คนหรือทางเดิน
- ไม่เห็นท่าทางคุกคาม
- ผู้ใช้ยังควรรักษาระยะและไม่เข้าใกล้

YELLOW:
- สุนัขอยู่ใกล้ทางเดิน ทางเข้า หรือพื้นที่คนสัญจร
- มีสุนัขหลายตัว หรือข้อมูลภาพไม่ชัดเจน
- สุนัขจ้อง ยืนระวัง เห่า หรือมีท่าทีตื่นตัว
- มีแม่สุนัขกับลูกสุนัข
- ผู้ใช้ควรเปลี่ยนเส้นทาง รักษาระยะ และไม่วิ่ง

RED:
- เห็นการวิ่งไล่ ล้อม เข้าประชิด หรือพุ่งเข้าใส่
- เห็นฟัน แยกเขี้ยว คำราม หรือท่าทางโจมตีชัดเจน
- มีการกัดหรือกำลังเผชิญหน้าโดยตรง
- เป็นฝูงที่กำลังปิดทางหรือคุกคามบุคคล
- มีอันตรายเฉียบพลันที่เห็นได้จากภาพ

กฎสำคัญ:
1. อย่าระบุสายพันธุ์เพื่อสรุปว่านิสัยดุ
2. อย่าทายพฤติกรรมที่ไม่ปรากฏในภาพ
3. ภาพนิ่งไม่สามารถยืนยันเสียงเห่าหรือการวิ่งไล่ได้
   เว้นแต่มีหลักฐานภาพต่อเนื่องหลายภาพรองรับ
4. ถ้าภาพไม่ชัด เห็นสุนัขไม่ครบ หรือไม่มั่นใจ
   ให้เลือก YELLOW และ needs_human_review=true
5. RED ต้องมีหลักฐานอันตรายที่มองเห็นได้ชัดเจน
6. ถ้าไม่พบสุนัข ให้เลือก YELLOW,
   dog_count=0 และ needs_human_review=true
7. คำแนะนำต้องสั้น ปฏิบัติได้ และไม่แนะนำให้จับ
   ให้อาหาร ไล่ ตะโกน หรือทำร้ายสุนัข
8. ตอบเป็นภาษาไทย ยกเว้นค่าที่กำหนดเป็นภาษาอังกฤษ

ส่งกลับเป็น JSON เท่านั้นตามโครงสร้างนี้:

{
  "risk_color": "green | yellow | red",
  "risk_score": 1,
  "dog_count": 0,
  "dog_count_description": "คำอธิบายจำนวนสุนัข",
  "observed_behavior": "พฤติกรรมหรือท่าทางที่มองเห็น",
  "visible_risk_factors": "ปัจจัยเสี่ยงที่มองเห็น",
  "recommendation": "คำแนะนำผู้ใช้งาน",
  "confidence": "low | medium | high",
  "image_quality": "clear | partly_clear | unclear",
  "needs_human_review": false
}

risk_score:
1 = GREEN
2 = YELLOW
3 = RED
"""

    content = [
        {
            "type": "input_text",
            "text": assessment_prompt,
        }
    ]

    for image_data in processed_images:
        content.append(
            {
                "type": "input_image",
                "image_url": bytes_to_data_url(
                    image_data["bytes"],
                    image_data["mime_type"],
                ),
                "detail": "high",
            }
        )

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    parsed = extract_json_object(response.output_text)
    return normalize_analysis(parsed)


# =========================================================
# GITHUB CSV
# =========================================================

def github_headers() -> dict:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_file_url() -> str:
    return (
        f"https://api.github.com/repos/"
        f"{GITHUB_OWNER}/{GITHUB_REPO}/contents/{GITHUB_CSV_PATH}"
    )


def read_github_csv() -> tuple[list[dict], str | None]:
    """
    คืนค่า:
    - รายการแถวเดิม
    - SHA ของไฟล์เดิม

    ถ้ายังไม่มีไฟล์ คืน rows=[] และ sha=None
    """

    response = requests.get(
        github_file_url(),
        headers=github_headers(),
        params={"ref": GITHUB_BRANCH},
        timeout=30,
    )

    if response.status_code == 404:
        return [], None

    response.raise_for_status()

    payload = response.json()
    encoded_content = payload.get("content", "").replace("\n", "")

    decoded_bytes = base64.b64decode(encoded_content)

    # utf-8-sig รองรับกรณีไฟล์เดิมมี BOM
    decoded_text = decoded_bytes.decode("utf-8-sig")

    if not decoded_text.strip():
        return [], payload.get("sha")

    reader = csv.DictReader(io.StringIO(decoded_text))
    rows = list(reader)

    return rows, payload.get("sha")


def rows_to_csv_bytes(rows: list[dict]) -> bytes:
    output = io.StringIO()

    writer = csv.DictWriter(
        output,
        fieldnames=CSV_FIELDS,
        extrasaction="ignore",
        lineterminator="\n",
    )

    writer.writeheader()

    for row in rows:
        safe_row = {
            field: row.get(field, "")
            for field in CSV_FIELDS
        }
        writer.writerow(safe_row)

    # utf-8-sig ช่วยให้เปิดภาษาไทยใน Excel ได้ดีขึ้น
    return output.getvalue().encode("utf-8-sig")


def append_report_to_github(
    report_row: dict,
    max_attempts: int = 4,
) -> None:
    required = {
        "GITHUB_TOKEN": GITHUB_TOKEN,
        "GITHUB_OWNER": GITHUB_OWNER,
        "GITHUB_REPO": GITHUB_REPO,
    }

    missing = [
        key for key, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "ยังไม่ได้ตั้งค่า: " + ", ".join(missing)
        )

    last_error = None

    for attempt in range(max_attempts):
        try:
            rows, current_sha = read_github_csv()
            rows.append(report_row)

            csv_bytes = rows_to_csv_bytes(rows)
            encoded_csv = base64.b64encode(csv_bytes).decode("utf-8")

            body = {
                "message": (
                    f"Add stray dog report "
                    f"{report_row['report_id']}"
                ),
                "content": encoded_csv,
                "branch": GITHUB_BRANCH,
            }

            if current_sha:
                body["sha"] = current_sha

            response = requests.put(
                github_file_url(),
                headers=github_headers(),
                json=body,
                timeout=30,
            )

            if response.status_code in (200, 201):
                return

            # มีผู้ใช้คนอื่นเขียนไฟล์พร้อมกัน
            # อ่านไฟล์ล่าสุดแล้วลองใหม่
            if response.status_code == 409:
                last_error = RuntimeError(
                    "GitHub CSV มีการเปลี่ยนแปลงพร้อมกัน"
                )
                time.sleep(0.8 * (attempt + 1))
                continue

            try:
                error_detail = response.json()
            except ValueError:
                error_detail = response.text

            raise RuntimeError(
                f"GitHub API error {response.status_code}: "
                f"{error_detail}"
            )

        except requests.RequestException as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))

    raise RuntimeError(
        f"บันทึก GitHub ไม่สำเร็จหลังลอง {max_attempts} ครั้ง: "
        f"{last_error}"
    )


# =========================================================
# MAP
# =========================================================

def display_location_map(
    latitude: float,
    longitude: float,
):
    map_data = [
        {
            "latitude": latitude,
            "longitude": longitude,
            "label": "ท่านอยู่ที่นี่",
        }
    ]

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position="[longitude, latitude]",
        get_radius=14,
        radius_min_pixels=9,
        radius_max_pixels=18,
        get_fill_color=[220, 30, 30, 230],
        get_line_color=[255, 255, 255],
        line_width_min_pixels=3,
        stroked=True,
        filled=True,
        pickable=True,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=map_data,
        get_position="[longitude, latitude]",
        get_text="label",
        get_size=18,
        get_color=[20, 20, 20],
        get_text_anchor="'middle'",
        get_alignment_baseline="'bottom'",
        get_pixel_offset=[0, -18],
    )

    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=17,
        pitch=0,
        bearing=0,
    )

    deck = pdk.Deck(
        layers=[point_layer, text_layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip={
            "html": (
                "<b>ท่านอยู่ที่นี่</b><br/>"
                "Latitude: {latitude}<br/>"
                "Longitude: {longitude}"
            ),
            "style": {
                "backgroundColor": "white",
                "color": "black",
            },
        },
    )

    st.pydeck_chart(
        deck,
        width="stretch",
        height=430,
    )


# =========================================================
# RESULT DISPLAY
# =========================================================

def display_analysis_result(result: dict):
    risk = RISK_DISPLAY[result["risk_color"]]

    st.markdown(
        f"""
        <div style="
            background:{risk['background']};
            border:3px solid {risk['border']};
            border-radius:18px;
            padding:22px;
            margin-top:12px;
            margin-bottom:18px;
        ">
            <div style="font-size:34px;font-weight:800;">
                {risk['icon']} {risk['thai']}
            </div>
            <div style="font-size:22px;font-weight:700;">
                {risk['title']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "จำนวนสุนัขที่ประเมินได้",
            result["dog_count"],
        )

    with col2:
        st.metric(
            "ระดับความเสี่ยง",
            f"{result['risk_score']} / 3",
        )

    with col3:
        confidence_display = {
            "low": "ต่ำ",
            "medium": "ปานกลาง",
            "high": "สูง",
        }.get(result["confidence"], result["confidence"])

        st.metric(
            "ความมั่นใจของ AI",
            confidence_display,
        )

    st.subheader("สิ่งที่ AI มองเห็น")
    st.write(
        result["observed_behavior"]
        or "ไม่สามารถอธิบายพฤติกรรมได้ชัดเจน"
    )

    st.subheader("ปัจจัยที่ใช้ประเมิน")
    st.write(
        result["visible_risk_factors"]
        or "ข้อมูลจากภาพยังมีจำกัด"
    )

    st.subheader("คำแนะนำ")
    st.info(
        result["recommendation"]
        or "รักษาระยะห่างและหลีกเลี่ยงการเข้าใกล้สุนัข"
    )

    if result["needs_human_review"]:
        st.warning(
            "ภาพหรือสถานการณ์นี้ควรได้รับการตรวจสอบเพิ่มเติม "
            "AI อาจประเมินได้ไม่ครบจากภาพนิ่ง"
        )


# =========================================================
# RESET
# =========================================================

def reset_report():
    st.session_state.analysis_result = None
    st.session_state.saved_report_id = None

    camera_keys = [
        "dog_camera_1",
        "dog_camera_2",
        "dog_camera_3",
        "dog_uploads",
    ]

    for key in camera_keys:
        if key in st.session_state:
            del st.session_state[key]


# =========================================================
# MAIN UI
# =========================================================

st.title("🐕 รายงานสุนัขในวิทยาเขต")
st.caption(
    "ระบุตำแหน่ง ถ่ายภาพ และให้ AI ช่วยประเมินความเสี่ยงเบื้องต้น"
)

st.warning(
    "ระบบนี้เป็นเครื่องมือช่วยสังเกตจากภาพ "
    "ไม่สามารถยืนยันอารมณ์หรือพฤติกรรมในอนาคตของสุนัขได้ "
    "หากมีอันตรายเฉียบพลันให้ถอยห่างและติดต่อเจ้าหน้าที่ทันที"
)


# =========================================================
# STEP 1: LOCATION
# =========================================================

st.header("1. ระบุตำแหน่งของท่าน")

location = streamlit_geolocation()

if isinstance(location, dict):
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    accuracy = location.get("accuracy")

    if latitude is not None and longitude is not None:
        st.session_state.latitude = float(latitude)
        st.session_state.longitude = float(longitude)

        if accuracy is not None:
            st.session_state.location_accuracy = float(accuracy)

if (
    st.session_state.latitude is not None
    and st.session_state.longitude is not None
):
    display_location_map(
        st.session_state.latitude,
        st.session_state.longitude,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Latitude",
            f"{st.session_state.latitude:.6f}",
        )

    with col2:
        st.metric(
            "Longitude",
            f"{st.session_state.longitude:.6f}",
        )

    with col3:
        if st.session_state.location_accuracy is not None:
            st.metric(
                "ความคลาดเคลื่อน",
                f"{st.session_state.location_accuracy:.1f} เมตร",
            )
        else:
            st.metric("ความคลาดเคลื่อน", "ไม่ทราบ")

else:
    st.info(
        "กดปุ่มรับตำแหน่งด้านบน และเลือก Allow หรืออนุญาต"
    )


# =========================================================
# STEP 2: IMAGES
# =========================================================

st.header("2. ถ่ายภาพหรือเลือกภาพสุนัข")

capture_tab, upload_tab = st.tabs(
    [
        "📷 เปิดกล้องถ่ายภาพ",
        "🖼️ Upload ภาพ",
    ]
)

camera_files = []

with capture_tab:
    st.caption(
        "สามารถถ่ายได้สูงสุด 3 ภาพ "
        "ภาพหลายมุมช่วยให้ประเมินบริบทได้ดีขึ้น"
    )

    camera_1 = st.camera_input(
        "ภาพที่ 1",
        key="dog_camera_1",
    )

    camera_2 = st.camera_input(
        "ภาพที่ 2",
        key="dog_camera_2",
    )

    camera_3 = st.camera_input(
        "ภาพที่ 3",
        key="dog_camera_3",
    )

    camera_files = [
        image_file
        for image_file in [camera_1, camera_2, camera_3]
        if image_file is not None
    ]


uploaded_files = []

with upload_tab:
    uploaded_files = st.file_uploader(
        "เลือกภาพได้สูงสุด 3 ภาพ",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="dog_uploads",
        help="รองรับ JPG, JPEG, PNG และ WEBP",
    )

    if len(uploaded_files) > 3:
        st.error(
            "เลือกภาพเกิน 3 ภาพ "
            "ระบบจะยังไม่อนุญาตให้วิเคราะห์"
        )


# ใช้ภาพจากทั้งสองแหล่งรวมกัน
selected_files = camera_files + uploaded_files

if len(selected_files) > 3:
    st.error(
        f"ขณะนี้มีภาพรวม {len(selected_files)} ภาพ "
        "กรุณาให้เหลือไม่เกิน 3 ภาพ"
    )

elif selected_files:
    st.success(
        f"เลือกภาพแล้ว {len(selected_files)} ภาพ"
    )

    preview_columns = st.columns(len(selected_files))

    for index, image_file in enumerate(selected_files):
        with preview_columns[index]:
            st.image(
                image_file,
                caption=f"ภาพที่ {index + 1}",
                width="stretch",
            )


# =========================================================
# STEP 3: ANALYZE
# =========================================================

st.header("3. ให้ GPT ประเมินความเสี่ยง")

can_analyze = (
    st.session_state.latitude is not None
    and st.session_state.longitude is not None
    and 1 <= len(selected_files) <= 3
)

if st.button(
    "🔍 วิเคราะห์ภาพสุนัข",
    type="primary",
    disabled=not can_analyze,
    width="stretch",
):
    st.session_state.analysis_result = None
    st.session_state.saved_report_id = None

    try:
        with st.spinner(
            "กำลังเตรียมภาพและวิเคราะห์ความเสี่ยง..."
        ):
            processed_images = [
                prepare_image(image_file)
                for image_file in selected_files
            ]

            analysis_result = analyze_dog_images(
                processed_images
            )

            analysis_result["image_count"] = len(
                processed_images
            )

            st.session_state.analysis_result = analysis_result

    except Exception as exc:
        st.error(f"วิเคราะห์ภาพไม่สำเร็จ: {exc}")


if not can_analyze:
    missing_items = []

    if (
        st.session_state.latitude is None
        or st.session_state.longitude is None
    ):
        missing_items.append("อนุญาตการใช้ตำแหน่ง")

    if not selected_files:
        missing_items.append("ถ่ายหรือ Upload อย่างน้อย 1 ภาพ")

    if len(selected_files) > 3:
        missing_items.append("ลดจำนวนภาพให้เหลือไม่เกิน 3 ภาพ")

    if missing_items:
        st.caption(
            "ก่อนวิเคราะห์ กรุณา: " + " และ ".join(missing_items)
        )


# =========================================================
# STEP 4: RESULT AND SAVE
# =========================================================

if st.session_state.analysis_result:
    st.header("4. ผลการประเมิน")

    result = st.session_state.analysis_result
    display_analysis_result(result)

    now_bkk = datetime.now(BKK_TZ)

    st.caption(
        "เวลาที่จะแนบกับรายงาน: "
        f"{now_bkk.strftime('%d/%m/%Y %H:%M:%S')} น. "
        "(Asia/Bangkok)"
    )

    if st.session_state.saved_report_id:
        st.success(
            "บันทึกข้อมูลเรียบร้อยแล้ว "
            f"รหัสรายงาน: {st.session_state.saved_report_id}"
        )

    else:
        save_col, reset_col = st.columns([3, 1])

        with save_col:
            save_clicked = st.button(
                "💾 ยืนยันและบันทึกลง GitHub CSV",
                type="primary",
                width="stretch",
            )

        with reset_col:
            reset_clicked = st.button(
                "เริ่มรายงานใหม่",
                width="stretch",
            )

        if reset_clicked:
            reset_report()
            st.rerun()

        if save_clicked:
            report_id = (
                "DOG-"
                + now_bkk.strftime("%Y%m%d-%H%M%S")
                + "-"
                + uuid.uuid4().hex[:6].upper()
            )

            report_row = {
                "report_id": report_id,
                "timestamp_bkk": now_bkk.isoformat(
                    timespec="seconds"
                ),
                "date_bkk": now_bkk.strftime("%Y-%m-%d"),
                "time_bkk": now_bkk.strftime("%H:%M:%S"),
                "latitude": (
                    f"{st.session_state.latitude:.7f}"
                ),
                "longitude": (
                    f"{st.session_state.longitude:.7f}"
                ),
                "location_accuracy_m": (
                    ""
                    if st.session_state.location_accuracy is None
                    else f"{st.session_state.location_accuracy:.1f}"
                ),
                "image_count": result["image_count"],
                "risk_color": result["risk_color"],
                "risk_level_th": result["risk_level_th"],
                "risk_score": result["risk_score"],
                "dog_count": result["dog_count"],
                "dog_count_description": (
                    result["dog_count_description"]
                ),
                "observed_behavior": (
                    result["observed_behavior"]
                ),
                "visible_risk_factors": (
                    result["visible_risk_factors"]
                ),
                "recommendation": result["recommendation"],
                "confidence": result["confidence"],
                "image_quality": result["image_quality"],
                "needs_human_review": (
                    result["needs_human_review"]
                ),
                "model": OPENAI_MODEL,
            }

            try:
                with st.spinner(
                    "กำลังบันทึกข้อมูลลง GitHub..."
                ):
                    append_report_to_github(report_row)

                st.session_state.saved_report_id = report_id
                st.success(
                    "บันทึกข้อมูลสำเร็จ โดยไม่ได้บันทึกภาพถ่าย"
                )
                st.rerun()

            except Exception as exc:
                st.error(f"บันทึกข้อมูลไม่สำเร็จ: {exc}")


# =========================================================
# PRIVACY NOTICE
# =========================================================

st.divider()

with st.expander("🔐 การใช้ข้อมูลและความเป็นส่วนตัว"):
    st.write(
        """
        - ระบบบันทึกเวลา พิกัด จำนวนภาพ และผลวิเคราะห์ลง GitHub CSV
        - ระบบไม่เขียนไฟล์ภาพลง GitHub
        - ภาพจะถูกส่งให้ OpenAI API เพื่อวิเคราะห์ในขณะใช้งาน
        - ไม่ควรถ่ายภาพใบหน้า ป้ายทะเบียน บัตรประจำตัว
          หรือข้อมูลที่ระบุตัวบุคคลได้
        - ตำแหน่งจากโทรศัพท์อาจมีความคลาดเคลื่อน
        - ผลจาก AI ควรใช้เป็นข้อมูลช่วยตัดสินใจ
          และไม่แทนการประเมินของเจ้าหน้าที่
        """
    )