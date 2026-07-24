import base64
import csv
import io
import math
import runpy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pydeck as pdk
import requests
import streamlit as st
from streamlit_geolocation import streamlit_geolocation


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="KU KPS Stray Dog Safety",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CONSTANTS
# =========================================================

BKK_TZ = ZoneInfo("Asia/Bangkok")

WATCH_RADIUS_M = 100
REPORT_AGE_HOURS = 12
ARROW_DISTANCE_M = 55

EARTH_RADIUS_M = 6_371_000

RISK_CONFIG = {
    "green": {
        "thai": "สีเขียว",
        "icon": "🟢",
        "title": "ความเสี่ยงต่ำ",
        "rgba": [35, 180, 80, 72],
        "solid_rgba": [35, 180, 80, 230],
        "weight": 0.50,
        "cloud_radius": 22,
    },
    "yellow": {
        "thai": "สีเหลือง",
        "icon": "🟡",
        "title": "ควรระมัดระวัง",
        "rgba": [255, 193, 7, 86],
        "solid_rgba": [245, 166, 35, 235],
        "weight": 2.00,
        "cloud_radius": 27,
    },
    "red": {
        "thai": "สีแดง",
        "icon": "🔴",
        "title": "ความเสี่ยงสูง",
        "rgba": [220, 45, 55, 98],
        "solid_rgba": [205, 35, 45, 240],
        "weight": 4.00,
        "cloud_radius": 34,
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

if "watch_latitude" not in st.session_state:
    st.session_state.watch_latitude = None

if "watch_longitude" not in st.session_state:
    st.session_state.watch_longitude = None

if "watch_accuracy" not in st.session_state:
    st.session_state.watch_accuracy = None


# =========================================================
# BASIC GEOGRAPHIC FUNCTIONS
# =========================================================

def haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    คำนวณระยะห่างระหว่างพิกัดสองจุดเป็นเมตร
    """

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return EARTH_RADIUS_M * c


def destination_point(
    latitude: float,
    longitude: float,
    bearing_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    """
    หาพิกัดปลายทางจากจุดเริ่มต้น ทิศ และระยะทาง
    """

    angular_distance = distance_m / EARTH_RADIUS_M
    bearing = math.radians(bearing_deg)

    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1)
        * math.sin(angular_distance)
        * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing)
        * math.sin(angular_distance)
        * math.cos(lat1),
        math.cos(angular_distance)
        - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def bearing_from_vector(east: float, north: float) -> float:
    """
    เปลี่ยนเวกเตอร์ East/North เป็นองศาทิศทาง
    0 = เหนือ, 90 = ตะวันออก
    """

    bearing = math.degrees(math.atan2(east, north))
    return (bearing + 360) % 360


def direction_name_th(bearing: float) -> str:
    directions = [
        "ทิศเหนือ",
        "ทิศตะวันออกเฉียงเหนือ",
        "ทิศตะวันออก",
        "ทิศตะวันออกเฉียงใต้",
        "ทิศใต้",
        "ทิศตะวันตกเฉียงใต้",
        "ทิศตะวันตก",
        "ทิศตะวันตกเฉียงเหนือ",
    ]

    index = int((bearing + 22.5) // 45) % 8
    return directions[index]


# =========================================================
# GITHUB CSV READING
# =========================================================

def github_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    return headers


def github_file_url() -> str:
    return (
        f"https://api.github.com/repos/"
        f"{GITHUB_OWNER}/{GITHUB_REPO}/contents/{GITHUB_CSV_PATH}"
    )


@st.cache_data(ttl=30, show_spinner=False)
def read_reports_from_github() -> list[dict]:
    """
    อ่านข้อมูล CSV จาก GitHub

    cache 30 วินาที เพื่อลดจำนวนครั้งที่เรียก GitHub API
    """

    required = {
        "GITHUB_OWNER": GITHUB_OWNER,
        "GITHUB_REPO": GITHUB_REPO,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "ยังไม่ได้ตั้งค่า " + ", ".join(missing)
        )

    response = requests.get(
        github_file_url(),
        headers=github_headers(),
        params={"ref": GITHUB_BRANCH},
        timeout=30,
    )

    if response.status_code == 404:
        return []

    response.raise_for_status()

    payload = response.json()

    encoded_content = (
        payload.get("content", "")
        .replace("\n", "")
        .strip()
    )

    if not encoded_content:
        return []

    csv_bytes = base64.b64decode(encoded_content)
    csv_text = csv_bytes.decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(csv_text))
    return list(reader)


# =========================================================
# DATA PARSING
# =========================================================

def parse_bool(value) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def parse_bkk_timestamp(value: str) -> datetime | None:
    """
    รองรับ timestamp เช่น:
    2026-07-24T07:30:15+07:00
    2026-07-24 07:30:15
    """

    if not value:
        return None

    cleaned = str(value).strip()

    try:
        timestamp = datetime.fromisoformat(cleaned)

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=BKK_TZ)
        else:
            timestamp = timestamp.astimezone(BKK_TZ)

        return timestamp

    except ValueError:
        pass

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]

    for fmt in formats:
        try:
            timestamp = datetime.strptime(cleaned, fmt)
            return timestamp.replace(tzinfo=BKK_TZ)
        except ValueError:
            continue

    return None


def normalize_risk_color(value: str) -> str:
    cleaned = str(value).strip().lower()

    color_map = {
        "green": "green",
        "สีเขียว": "green",
        "เขียว": "green",
        "yellow": "yellow",
        "สีเหลือง": "yellow",
        "เหลือง": "yellow",
        "red": "red",
        "สีแดง": "red",
        "แดง": "red",
    }

    return color_map.get(cleaned, "yellow")


def get_active_nearby_reports(
    rows: list[dict],
    user_latitude: float,
    user_longitude: float,
) -> list[dict]:
    """
    เลือกเฉพาะ:
    - อายุข้อมูลไม่เกิน 12 ชั่วโมง
    - อยู่ภายใน 100 เมตรจากผู้ใช้
    """

    now_bkk = datetime.now(BKK_TZ)
    cutoff_time = now_bkk - timedelta(hours=REPORT_AGE_HOURS)

    active_reports = []

    for row in rows:
        try:
            latitude = float(row.get("latitude", ""))
            longitude = float(row.get("longitude", ""))
        except (TypeError, ValueError):
            continue

        timestamp = parse_bkk_timestamp(
            row.get("timestamp_bkk", "")
        )

        if timestamp is None:
            continue

        if timestamp < cutoff_time or timestamp > now_bkk:
            continue

        distance_m = haversine_distance_m(
            user_latitude,
            user_longitude,
            latitude,
            longitude,
        )

        if distance_m > WATCH_RADIUS_M:
            continue

        risk_color = normalize_risk_color(
            row.get("risk_color", "")
        )

        age_minutes = max(
            0,
            int((now_bkk - timestamp).total_seconds() / 60),
        )

        active_reports.append(
            {
                "report_id": row.get("report_id", ""),
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": timestamp,
                "timestamp_display": timestamp.strftime(
                    "%d/%m/%Y %H:%M"
                ),
                "age_minutes": age_minutes,
                "distance_m": round(distance_m, 1),
                "risk_color": risk_color,
                "risk_level_th": RISK_CONFIG[risk_color]["thai"],
                "risk_score": row.get("risk_score", ""),
                "dog_count": row.get("dog_count", ""),
                "observed_behavior": row.get(
                    "observed_behavior",
                    "",
                ),
                "visible_risk_factors": row.get(
                    "visible_risk_factors",
                    "",
                ),
                "recommendation": row.get(
                    "recommendation",
                    "",
                ),
                "confidence": row.get("confidence", ""),
                "needs_human_review": parse_bool(
                    row.get("needs_human_review", False)
                ),
            }
        )

    active_reports.sort(
        key=lambda item: (
            item["distance_m"],
            -RISK_CONFIG[item["risk_color"]]["weight"],
        )
    )

    return active_reports


# =========================================================
# COLORED CLOUDS
# =========================================================

def create_cloud_points(report: dict) -> list[dict]:
    """
    สร้างวงกลมโปร่งแสงซ้อนกันรอบพิกัด
    ทำให้ดูคล้าย Colored cloud
    """

    config = RISK_CONFIG[report["risk_color"]]
    base_radius = config["cloud_radius"]

    # จุดกลางและก้อนย่อยโดยรอบ
    cloud_pattern = [
        (0, 0, 1.20),
        (9, 0, 0.95),
        (-9, 0, 0.95),
        (0, 9, 0.90),
        (0, -9, 0.90),
        (7, 7, 0.82),
        (-7, 7, 0.82),
        (7, -7, 0.82),
        (-7, -7, 0.82),
    ]

    points = []

    for index, (east_m, north_m, scale) in enumerate(
        cloud_pattern
    ):
        distance = math.sqrt(east_m**2 + north_m**2)

        if distance == 0:
            latitude = report["latitude"]
            longitude = report["longitude"]
        else:
            bearing = bearing_from_vector(
                east=east_m,
                north=north_m,
            )

            latitude, longitude = destination_point(
                report["latitude"],
                report["longitude"],
                bearing,
                distance,
            )

        points.append(
            {
                "cloud_id": (
                    f"{report['report_id']}-{index}"
                ),
                "latitude": latitude,
                "longitude": longitude,
                "radius_m": base_radius * scale,
                "fill_color": config["rgba"],
                "risk_color": report["risk_color"],
                "risk_level_th": report["risk_level_th"],
                "distance_m": report["distance_m"],
                "age_minutes": report["age_minutes"],
                "timestamp_display": report[
                    "timestamp_display"
                ],
                "dog_count": report["dog_count"],
                "observed_behavior": report[
                    "observed_behavior"
                ],
                "recommendation": report["recommendation"],
            }
        )

    return points


# =========================================================
# SAFE DIRECTION CALCULATION
# =========================================================

def calculate_risk_at_candidate(
    candidate_latitude: float,
    candidate_longitude: float,
    reports: list[dict],
) -> float:
    """
    คะแนนยิ่งต่ำ ยิ่งอยู่ห่างจากรายงานความเสี่ยงมาก

    ใช้ inverse-distance weighting
    """

    total_risk = 0.0

    for report in reports:
        distance = haversine_distance_m(
            candidate_latitude,
            candidate_longitude,
            report["latitude"],
            report["longitude"],
        )

        weight = RISK_CONFIG[
            report["risk_color"]
        ]["weight"]

        # ป้องกันหารด้วยศูนย์
        effective_distance = max(distance, 5)

        total_risk += weight / (effective_distance**1.35)

    return total_risk


def choose_lower_risk_direction(
    user_latitude: float,
    user_longitude: float,
    reports: list[dict],
) -> dict | None:
    """
    ทดสอบ 16 ทิศรอบตัวผู้ใช้
    และเลือกทิศที่มีคะแนนความเสี่ยงต่ำที่สุด

    ไม่ใช่การคำนวณเส้นทางตามถนน
    """

    if not reports:
        return None

    candidates = []

    for bearing in range(0, 360, 22):
        endpoint_lat, endpoint_lon = destination_point(
            user_latitude,
            user_longitude,
            bearing,
            ARROW_DISTANCE_M,
        )

        risk_score = calculate_risk_at_candidate(
            endpoint_lat,
            endpoint_lon,
            reports,
        )

        candidates.append(
            {
                "bearing": float(bearing),
                "end_latitude": endpoint_lat,
                "end_longitude": endpoint_lon,
                "risk_score": risk_score,
            }
        )

    best = min(
        candidates,
        key=lambda candidate: candidate["risk_score"],
    )

    best["direction_th"] = direction_name_th(
        best["bearing"]
    )

    return best


def make_arrow_head(
    end_latitude: float,
    end_longitude: float,
    bearing: float,
) -> list[list[float]]:
    """
    สร้างหัวลูกศรเป็นรูปสามเหลี่ยม
    """

    back_left_bearing = (bearing + 150) % 360
    back_right_bearing = (bearing - 150) % 360

    left_lat, left_lon = destination_point(
        end_latitude,
        end_longitude,
        back_left_bearing,
        9,
    )

    right_lat, right_lon = destination_point(
        end_latitude,
        end_longitude,
        back_right_bearing,
        9,
    )

    return [
        [end_longitude, end_latitude],
        [left_lon, left_lat],
        [right_lon, right_lat],
    ]


# =========================================================
# MAP DISPLAY
# =========================================================

def display_watch_map(
    user_latitude: float,
    user_longitude: float,
    reports: list[dict],
    direction: dict | None,
):
    layers = []

    # -----------------------------------------------------
    # พื้นที่ 100 เมตรรอบผู้ใช้
    # -----------------------------------------------------

    radius_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[
            {
                "latitude": user_latitude,
                "longitude": user_longitude,
                "radius_m": WATCH_RADIUS_M,
            }
        ],
        get_position="[longitude, latitude]",
        get_radius="radius_m",
        radius_units="meters",
        get_fill_color=[70, 130, 180, 10],
        get_line_color=[70, 130, 180, 150],
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
        pickable=False,
    )

    layers.append(radius_layer)

    # -----------------------------------------------------
    # Colored clouds
    # -----------------------------------------------------

    cloud_points = []

    for report in reports:
        cloud_points.extend(create_cloud_points(report))

    if cloud_points:
        cloud_layer = pdk.Layer(
            "ScatterplotLayer",
            data=cloud_points,
            get_position="[longitude, latitude]",
            get_radius="radius_m",
            radius_units="meters",
            get_fill_color="fill_color",
            stroked=False,
            filled=True,
            pickable=True,
        )

        layers.append(cloud_layer)

        # จุดกลางของรายงาน
        report_points = []

        for report in reports:
            config = RISK_CONFIG[report["risk_color"]]

            report_points.append(
                {
                    **report,
                    "point_color": config["solid_rgba"],
                }
            )

        report_point_layer = pdk.Layer(
            "ScatterplotLayer",
            data=report_points,
            get_position="[longitude, latitude]",
            get_radius=5,
            radius_units="meters",
            radius_min_pixels=6,
            radius_max_pixels=11,
            get_fill_color="point_color",
            get_line_color=[255, 255, 255, 240],
            line_width_min_pixels=2,
            stroked=True,
            filled=True,
            pickable=True,
        )

        layers.append(report_point_layer)

    # -----------------------------------------------------
    # ลูกศรทิศลดความเสี่ยง
    # -----------------------------------------------------

    if direction:
        arrow_path_data = [
            {
                "path": [
                    [user_longitude, user_latitude],
                    [
                        direction["end_longitude"],
                        direction["end_latitude"],
                    ],
                ]
            }
        ]

        arrow_path_layer = pdk.Layer(
            "PathLayer",
            data=arrow_path_data,
            get_path="path",
            get_color=[30, 105, 210, 240],
            get_width=7,
            width_min_pixels=5,
            width_max_pixels=10,
            rounded=True,
            pickable=False,
        )

        layers.append(arrow_path_layer)

        arrow_head_data = [
            {
                "polygon": make_arrow_head(
                    direction["end_latitude"],
                    direction["end_longitude"],
                    direction["bearing"],
                )
            }
        ]

        arrow_head_layer = pdk.Layer(
            "PolygonLayer",
            data=arrow_head_data,
            get_polygon="polygon",
            get_fill_color=[30, 105, 210, 245],
            get_line_color=[255, 255, 255, 220],
            line_width_min_pixels=1,
            stroked=True,
            filled=True,
            pickable=False,
        )

        layers.append(arrow_head_layer)

        arrow_text_layer = pdk.Layer(
            "TextLayer",
            data=[
                {
                    "latitude": direction["end_latitude"],
                    "longitude": direction["end_longitude"],
                    "label": (
                        f"แนวทางลดความเสี่ยง\n"
                        f"{direction['direction_th']}"
                    ),
                }
            ],
            get_position="[longitude, latitude]",
            get_text="label",
            get_size=15,
            get_color=[15, 65, 150, 255],
            get_text_anchor="'middle'",
            get_alignment_baseline="'bottom'",
            get_pixel_offset=[0, -18],
        )

        layers.append(arrow_text_layer)

    # -----------------------------------------------------
    # จุดผู้ใช้
    # -----------------------------------------------------

    user_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[
            {
                "latitude": user_latitude,
                "longitude": user_longitude,
            }
        ],
        get_position="[longitude, latitude]",
        get_radius=7,
        radius_units="meters",
        radius_min_pixels=9,
        radius_max_pixels=15,
        get_fill_color=[20, 100, 235, 255],
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=3,
        stroked=True,
        filled=True,
        pickable=True,
    )

    layers.append(user_layer)

    user_text_layer = pdk.Layer(
        "TextLayer",
        data=[
            {
                "latitude": user_latitude,
                "longitude": user_longitude,
                "label": "ท่านอยู่ที่นี่",
            }
        ],
        get_position="[longitude, latitude]",
        get_text="label",
        get_size=18,
        get_color=[15, 40, 90, 255],
        get_text_anchor="'middle'",
        get_alignment_baseline="'bottom'",
        get_pixel_offset=[0, -16],
    )

    layers.append(user_text_layer)

    view_state = pdk.ViewState(
        latitude=user_latitude,
        longitude=user_longitude,
        zoom=18,
        pitch=0,
        bearing=0,
    )

    tooltip = {
        "html": """
        <b>{risk_level_th}</b><br/>
        ระยะจากท่าน: {distance_m} เมตร<br/>
        อายุข้อมูล: {age_minutes} นาที<br/>
        จำนวนสุนัข: {dog_count}<br/>
        เวลา: {timestamp_display}<br/>
        สิ่งที่พบ: {observed_behavior}<br/>
        คำแนะนำ: {recommendation}
        """,
        "style": {
            "backgroundColor": "white",
            "color": "black",
            "fontSize": "13px",
            "maxWidth": "340px",
        },
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="light",
        tooltip=tooltip,
    )

    st.pydeck_chart(
        deck,
        width="stretch",
        height=650,
    )


# =========================================================
# RISK SUMMARY
# =========================================================

def determine_current_risk(reports: list[dict]) -> str:
    if any(
        report["risk_color"] == "red"
        for report in reports
    ):
        return "red"

    if any(
        report["risk_color"] == "yellow"
        for report in reports
    ):
        return "yellow"

    if reports:
        return "green"

    return "none"


def display_risk_summary(
    reports: list[dict],
    direction: dict | None,
):
    current_risk = determine_current_risk(reports)

    if current_risk == "none":
        st.success(
            "🟢 ไม่พบรายงานสุนัขภายในรัศมี 100 เมตร "
            "ในช่วง 12 ชั่วโมงที่ผ่านมา"
        )
        st.caption(
            "ยังควรสังเกตสภาพแวดล้อมจริง "
            "เพราะอาจมีสุนัขที่ยังไม่มีผู้รายงาน"
        )
        return

    nearest = min(
        reports,
        key=lambda report: report["distance_m"],
    )

    highest_risk = max(
        reports,
        key=lambda report: RISK_CONFIG[
            report["risk_color"]
        ]["weight"],
    )

    config = RISK_CONFIG[current_risk]

    if current_risk == "red":
        st.error(
            f"{config['icon']} พบพื้นที่ความเสี่ยงสูง "
            f"ภายในรัศมี {WATCH_RADIUS_M} เมตร"
        )

    elif current_risk == "yellow":
        st.warning(
            f"{config['icon']} พบพื้นที่ที่ควรระมัดระวัง "
            f"ภายในรัศมี {WATCH_RADIUS_M} เมตร"
        )

    else:
        st.success(
            f"{config['icon']} รายงานที่พบอยู่ในระดับความเสี่ยงต่ำ"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "รายงานที่ยังมีผล",
            f"{len(reports)} จุด",
        )

    with col2:
        st.metric(
            "จุดใกล้ที่สุด",
            f"{nearest['distance_m']:.1f} เมตร",
        )

    with col3:
        st.metric(
            "ระดับสูงสุด",
            highest_risk["risk_level_th"],
        )

    if direction:
        st.info(
            f"➡️ แนวทางที่มีความเสี่ยงน้อยกว่าโดยประมาณ: "
            f"**{direction['direction_th']}**"
        )

    if current_risk == "red":
        st.markdown(
            """
            **การปฏิบัติขณะนี้**

            รักษาระยะ ไม่วิ่ง ไม่จ้องตา และไม่เดินเข้าไปในกลุ่มสุนัข  
            หันกลับหรือเลือกเส้นทางอื่นที่มองเห็นได้ชัดเจน
            """
        )

    elif current_risk == "yellow":
        st.markdown(
            """
            **การปฏิบัติขณะนี้**

            ชะลอการเดิน สังเกตสุนัข และเว้นระยะ  
            เลือกทางอื่นเมื่อสุนัขอยู่ใกล้ทางเดินหรือรวมกันหลายตัว
            """
        )

    else:
        st.markdown(
            """
            **การปฏิบัติขณะนี้**

            เดินต่อได้ด้วยความระมัดระวัง รักษาระยะ  
            และไม่เข้าใกล้ ไม่ให้อาหาร หรือสัมผัสสุนัข
            """
        )


# =========================================================
# WATCH PAGE
# =========================================================

def render_watch_page():
    st.title("🐕 ระวังสุนัข")
    st.caption(
        "แสดงรายงานภายในรัศมี 100 เมตร "
        "และย้อนหลังไม่เกิน 12 ชั่วโมง"
    )

    now_bkk = datetime.now(BKK_TZ)

    st.caption(
        "เวลาปัจจุบัน: "
        f"{now_bkk.strftime('%d/%m/%Y %H:%M:%S')} น. "
        "(Asia/Bangkok)"
    )

    location = streamlit_geolocation()

    if isinstance(location, dict):
        latitude = location.get("latitude")
        longitude = location.get("longitude")
        accuracy = location.get("accuracy")

        if latitude is not None and longitude is not None:
            st.session_state.watch_latitude = float(latitude)
            st.session_state.watch_longitude = float(longitude)

            if accuracy is not None:
                st.session_state.watch_accuracy = float(accuracy)

    if (
        st.session_state.watch_latitude is None
        or st.session_state.watch_longitude is None
    ):
        st.info(
            "กรุณากดปุ่มรับตำแหน่ง และเลือก Allow "
            "เพื่อดู Colored clouds รอบตัวท่าน"
        )
        st.stop()

    user_latitude = st.session_state.watch_latitude
    user_longitude = st.session_state.watch_longitude

    location_col1, location_col2, location_col3 = st.columns(3)

    with location_col1:
        st.metric(
            "Latitude",
            f"{user_latitude:.6f}",
        )

    with location_col2:
        st.metric(
            "Longitude",
            f"{user_longitude:.6f}",
        )

    with location_col3:
        if st.session_state.watch_accuracy is not None:
            st.metric(
                "ความคลาดเคลื่อน",
                f"{st.session_state.watch_accuracy:.1f} เมตร",
            )
        else:
            st.metric("ความคลาดเคลื่อน", "ไม่ทราบ")

    refresh_col, info_col = st.columns([1, 4])

    with refresh_col:
        if st.button(
            "🔄 Refresh",
            width="stretch",
        ):
            read_reports_from_github.clear()
            st.rerun()

    with info_col:
        st.caption(
            "ข้อมูลจาก GitHub จะถูกตรวจใหม่อัตโนมัติภายในประมาณ "
            "30 วินาที หรือกด Refresh เพื่ออ่านข้อมูลล่าสุดทันที"
        )

    try:
        with st.spinner(
            "กำลังอ่านข้อมูลสุนัขและคำนวณความเสี่ยง..."
        ):
            all_reports = read_reports_from_github()

            active_reports = get_active_nearby_reports(
                all_reports,
                user_latitude,
                user_longitude,
            )

            safe_direction = choose_lower_risk_direction(
                user_latitude,
                user_longitude,
                active_reports,
            )

    except Exception as exc:
        st.error(f"อ่านข้อมูลจาก GitHub ไม่สำเร็จ: {exc}")
        st.stop()

    display_risk_summary(
        active_reports,
        safe_direction,
    )

    display_watch_map(
        user_latitude,
        user_longitude,
        active_reports,
        safe_direction,
    )

    st.caption(
        "วงกลมเส้นขอบแสดงพื้นที่ 100 เมตรรอบตำแหน่งของท่าน "
        "และ Colored clouds แสดงตำแหน่งรายงานที่ยังมีอายุไม่เกิน "
        "12 ชั่วโมง"
    )

    st.warning(
        "ลูกศรเป็นเพียงทิศทางเบื้องต้นที่คำนวณจากตำแหน่งรายงาน "
        "ระบบยังไม่ทราบอาคาร รั้ว ถนน คูน้ำ รถ หรือสิ่งกีดขวาง "
        "โปรดตรวจเส้นทางจริงก่อนเดินต่อ"
    )

    if active_reports:
        with st.expander(
            f"ดูรายละเอียดรายงาน {len(active_reports)} จุด"
        ):
            for index, report in enumerate(
                active_reports,
                start=1,
            ):
                config = RISK_CONFIG[
                    report["risk_color"]
                ]

                st.markdown(
                    f"""
                    **{index}. {config['icon']} 
                    {report['risk_level_th']}**

                    ระยะจากท่าน: **{report['distance_m']} เมตร**  
                    อายุข้อมูล: **{report['age_minutes']} นาที**  
                    เวลารายงาน: {report['timestamp_display']}  
                    จำนวนสุนัข: {report['dog_count'] or 'ไม่ระบุ'}  
                    สิ่งที่พบ: {
                        report['observed_behavior']
                        or 'ไม่มีคำอธิบาย'
                    }  
                    คำแนะนำ: {
                        report['recommendation']
                        or 'รักษาระยะห่าง'
                    }
                    """
                )

                if index < len(active_reports):
                    st.divider()


# =========================================================
# SIDEBAR MENU
# =========================================================

with st.sidebar:
    st.title("🐕 Stray Dog Safety")

    page = st.radio(
        "เลือกการใช้งาน",
        options=[
            "ระวังสุนัข",
            "ถ่ายภาพสุนัข",
        ],
        index=0,
    )

    st.divider()

    if page == "ระวังสุนัข":
        st.info(
            "ดู Colored clouds และแนวทางลดความเสี่ยง "
            "จากรายงานย้อนหลัง 12 ชั่วโมง"
        )
    else:
        st.info(
            "ถ่ายหรือ Upload ภาพเพื่อเพิ่มข้อมูลสุนัข "
            "ลงในระบบ"
        )


# =========================================================
# PAGE ROUTER
# =========================================================

if page == "ระวังสุนัข":
    render_watch_page()

else:
    # เรียกโปรแกรมถ่ายภาพเดิมที่ทดสอบสำเร็จแล้ว
    runpy.run_path(
        "dog_report_page.py",
        run_name="__main__",
    )