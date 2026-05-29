
# 폐암 환자 군집 분석 머신러닝 모델 (K-means, K=4)

import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# ─────────────────────────────────────────────────────────────
# 현재 폴더 경로
# ─────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# GitHub 포함 malgun.ttf 사용
# ─────────────────────────────────────────────────────────────
font_path = os.path.join(APP_DIR, "fonts", "malgun.ttf")

fm.fontManager.addfont(font_path)

font_name = fm.FontProperties(
    fname=font_path
).get_name()

plt.rc("font", family=font_name)

mpl.rcParams["font.family"] = font_name
mpl.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────────────────────
# Streamlit 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="폐암 환자 군집 분석",
    page_icon="🫁",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────
# CSS 스타일
# ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>

    html, body, [class*="css"], .stApp {{
        font-family: '{font_name}', sans-serif !important;
    }}

    .stApp {{
        background: linear-gradient(
            180deg,
            #eef5ff 0%,
            #f7fbff 45%,
            #ffffff 100%
        );
    }}

    .block-container {{
        max-width: 880px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }}

    .hero {{
        background: linear-gradient(
            135deg,
            #2563eb 0%,
            #06b6d4 100%
        );

        border-radius: 22px;

        padding: 34px 38px;

        color: #ffffff;

        box-shadow:
            0 14px 34px rgba(37, 99, 235, 0.28);

        margin-bottom: 22px;
    }}

    .hero .badge {{
        display: inline-block;

        background:
            rgba(255, 255, 255, 0.18);

        border:
            1px solid rgba(255, 255, 255, 0.35);

        border-radius: 999px;

        padding: 4px 14px;

        font-size: 0.78rem;
        font-weight: 700;

        margin-bottom: 14px;
    }}

    .hero h1 {{
        margin: 0;

        font-size: 1.9rem;
        font-weight: 900;

        line-height: 1.3;
    }}

    .hero p {{
        margin-top: 10px;

        font-size: 1rem;

        color:
            rgba(255,255,255,0.92);
    }}

    .section-title {{
        font-size: 1.05rem;
        font-weight: 800;

        color: #0f2c5c;

        margin: 26px 0 6px 0;

        padding-left: 12px;

        border-left: 5px solid #2563eb;
    }}

    .stButton > button {{
        background: linear-gradient(
            135deg,
            #2563eb 0%,
            #06b6d4 100%
        );

        color: white;

        border: none;

        border-radius: 14px;

        padding: 0.7rem 1.4rem;

        font-weight: 800;

        width: 100%;

        transition: 0.2s;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
    }}

    .stAlert {{
        border-radius: 14px;
    }}

    [data-testid="stSidebar"] {{
        background: #ffffff;
        border-right: 1px solid #e3edfa;
    }}

    [data-testid="stSlider"] [role="slider"] {{
        background-color: #2563eb !important;
        border-color: #2563eb !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">

        <span class="badge">
            🫁 K-MEANS · K=4
        </span>

        <h1>
            폐암 환자 군집 분석 머신러닝 모델
        </h1>

        <p>
            나이 · 흡연량 · 음주량을 입력하면
            어느 군집에 속하는지 분류합니다.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header(
    "머신러닝 모델 설계 실습 (K-means 군집화)"
)

# ─────────────────────────────────────────────────────────────
# CSV 로드 함수
# ─────────────────────────────────────────────────────────────
def load_csv_robust(path):

    for enc in (
        "utf-8",
        "utf-8-sig",
        "cp949"
    ):

        try:
            return pd.read_csv(
                path,
                encoding=enc
            )

        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)

# ─────────────────────────────────────────────────────────────
# 모델 재학습 함수
# ─────────────────────────────────────────────────────────────
def rebuild_from_csv(df):

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X = df[["나이", "흡연량", "음주량"]]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    return model, scaler

# ─────────────────────────────────────────────────────────────
# 군집 설명 생성
# ─────────────────────────────────────────────────────────────
def build_cluster_info(df):

    info = {}

    has_result = "결과" in df.columns

    for c in sorted(df["cluster"].unique()):

        sub = df[df["cluster"] == c]

        n = len(sub)

        if has_result:

            rate = sub["결과"].mean() * 100

            if rate >= 80:
                tag = "폐암 고위험군"

            elif rate == 0:
                tag = "건강군"

            elif rate <= 20:
                tag = "저위험 건강군"

            else:
                tag = "중간 위험군"

            info[int(c)] = (
                f"{tag} ({n}명, 폐암 비율 {rate:.0f}%)"
            )

        else:

            info[int(c)] = f"군집 {c} ({n}명)"

    return info

# ─────────────────────────────────────────────────────────────
# 모델 / 데이터 로드
# ─────────────────────────────────────────────────────────────
model = None
scaler = None
df = None

model_path = os.path.join(
    APP_DIR,
    "lung_model.pkl"
)

scaler_path = os.path.join(
    APP_DIR,
    "scaler.pkl"
)

df_path = os.path.join(
    APP_DIR,
    "lung.csv"
)

# CSV 로드
try:

    df = load_csv_robust(df_path)

except Exception as e:

    st.error(
        f"lung.csv 로드 실패: {e}"
    )

    st.stop()

# pkl 로드
try:

    model = joblib.load(model_path)

    scaler = joblib.load(scaler_path)

    _ = scaler.transform(
        df[["나이", "흡연량", "음주량"]].head(1)
    )

    st.success("저장된 모델 로드 성공!")

except Exception:

    st.info(
        "저장 모델 사용 실패 → CSV로 재학습합니다."
    )

    model, scaler = rebuild_from_csv(df)

    X = df[["나이", "흡연량", "음주량"]]

    df["cluster"] = model.predict(
        scaler.transform(X)
    )

    st.success("재학습 완료!")

# ─────────────────────────────────────────────────────────────
# 사이드바 군집 정보
# ─────────────────────────────────────────────────────────────
cluster_info = build_cluster_info(df)

sidebar_lines = ["## 군집 정보"]

for c in sorted(cluster_info.keys()):

    sidebar_lines.append(
        f"- {c}번: {cluster_info[c]}"
    )

st.sidebar.markdown(
    "\n".join(sidebar_lines)
)

# ─────────────────────────────────────────────────────────────
# 입력 UI
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title">📋 환자 정보 입력</div>',
    unsafe_allow_html=True
)

age = st.slider(
    "나이",
    10.0,
    90.0,
    40.0
)

smoking = st.slider(
    "흡연량",
    0.0,
    40.0,
    10.0
)

drinking = st.slider(
    "음주량",
    0.0,
    10.0,
    3.0
)

new_patient = pd.DataFrame(
    [[age, smoking, drinking]],
    columns=[
        "나이",
        "흡연량",
        "음주량"
    ]
)

# ─────────────────────────────────────────────────────────────
# 예측
# ─────────────────────────────────────────────────────────────
if st.button("군집 예측하기"):

    try:

        new_patient_scaled = scaler.transform(
            new_patient
        )

        pred_cluster = int(
            model.predict(
                new_patient_scaled
            )[0]
        )

        st.success(
            f"이 환자는 "
            f"{pred_cluster}번 군집입니다."
        )

        st.info(
            cluster_info.get(
                pred_cluster,
                "알 수 없음"
            )
        )

        # ─────────────────────────────────────
        # 그래프 1
        # ─────────────────────────────────────
        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        fig.patch.set_facecolor("#ffffff")

        ax.set_facecolor("#f6faff")

        ax.scatter(
            df["나이"],
            df["흡연량"],
            c=df["cluster"],
            cmap="viridis",
            alpha=0.6,
            s=55
        )

        ax.scatter(
            age,
            smoking,
            c="#e11d48",
            s=340,
            marker="X",
            edgecolors="white",
            linewidths=2.2
        )

        ax.set_xlabel("나이")
        ax.set_ylabel("흡연량")

        ax.set_title(
            f"폐암 환자 군집 분석 ({pred_cluster}번 군집)"
        )

        ax.grid(True, linestyle="--", alpha=0.35)

        st.pyplot(fig)

        plt.close(fig)

        # ─────────────────────────────────────
        # 그래프 2
        # ─────────────────────────────────────
        fig2, ax2 = plt.subplots(
            figsize=(10, 7)
        )

        fig2.patch.set_facecolor("#ffffff")

        ax2.set_facecolor("#f6faff")

        ax2.scatter(
            df["나이"],
            df["음주량"],
            c=df["cluster"],
            cmap="viridis",
            alpha=0.6,
            s=55
        )

        ax2.scatter(
            age,
            drinking,
            c="#e11d48",
            s=340,
            marker="X",
            edgecolors="white",
            linewidths=2.2
        )

        ax2.set_xlabel("나이")
        ax2.set_ylabel("음주량")

        ax2.set_title("나이 vs 음주량")

        ax2.grid(True, linestyle="--", alpha=0.35)

        st.pyplot(fig2)

        plt.close(fig2)

    except Exception as e:

        st.error(
            f"예측 실패: {e}"
        )
