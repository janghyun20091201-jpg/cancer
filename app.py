# 폐암 환자 군집 분석 머신러닝 모델 (K-means, K=4)
#
# 동작 방식:
#   1) lung_model.pkl + scaler.pkl 로드 시도
#   2) 실패하면 (예: sklearn 버전 차이) lung.csv 로 자동 재학습
#
# 필요한 파일: lung.csv  (lung_model.pkl, scaler.pkl 은 있으면 사용)

import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# ─────────────────────────────────────────────────────────────
# 한글 폰트 설정 (Windows 한글 깨짐 해결)
# ─────────────────────────────────────────────────────────────
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕
    font_name = fm.FontProperties(fname=font_path).get_name()
    mpl.rc("font", family=font_name)

    # 마이너스 깨짐 방지
    mpl.rcParams["axes.unicode_minus"] = False

except Exception:
    pass

# koreanize_matplotlib 있으면 추가 적용
try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    pass


# app.py 가 있는 폴더 기준 절대경로
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def load_csv_robust(path):
    """UTF-8 / UTF-8-sig / CP949 순서로 시도 (한글 윈도우 호환)"""
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def rebuild_from_csv(df):
    """lung.csv 의 데이터로 scaler + KMeans 를 다시 학습.
    노트북과 동일한 하이퍼파라미터(K=4, random_state=42) 사용."""
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


def build_cluster_info(df):
    """군집별 인원수 + 폐암 양성 비율 자동 계산"""

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
# Streamlit 페이지 설정 (디자인)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="폐암 환자 군집 분석",
    page_icon="🫁",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────
# 전역 스타일 (디자인)
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Noto Sans KR', -apple-system, sans-serif;
    }

    /* 전체 배경 */
    .stApp {
        background: linear-gradient(180deg, #eef5ff 0%, #f7fbff 45%, #ffffff 100%);
    }

    /* 본문 폭 / 여백 */
    .block-container {
        max-width: 880px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    /* 히어로 헤더 */
    .hero {
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
        border-radius: 22px;
        padding: 34px 38px;
        color: #ffffff;
        box-shadow: 0 14px 34px rgba(37, 99, 235, 0.28);
        margin-bottom: 22px;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 14px;
    }
    .hero h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 900;
        line-height: 1.25;
    }
    .hero p {
        margin: 10px 0 0 0;
        font-size: 0.98rem;
        font-weight: 400;
        color: rgba(255, 255, 255, 0.92);
        line-height: 1.6;
    }

    /* 섹션 제목 */
    .section-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0f2c5c;
        margin: 26px 0 6px 0;
        padding-left: 12px;
        border-left: 5px solid #2563eb;
    }

    /* 버튼 */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
        color: #ffffff;
        border: none;
        border-radius: 14px;
        padding: 0.65rem 1.4rem;
        font-weight: 800;
        font-size: 1rem;
        width: 100%;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.32);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(37, 99, 235, 0.42);
        color: #ffffff;
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* 알림 박스 라운드 */
    .stAlert {
        border-radius: 14px;
    }

    /* 사이드바 */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e3edfa;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* 슬라이더 강조색 */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# 히어로 헤더 (디자인)
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <span class="badge">🫁 K-MEANS · K=4</span>
        <h1>폐암 환자 군집 분석 머신러닝 모델</h1>
        <p>나이 · 흡연량 · 음주량을 입력하면 어느 군집에 속하는지 분류합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header(
    "머신러닝 모델 설계 실습 (K-means 군집화)"
)


# ─────────────────────────────────────────────────────────────
# 모델 / 스케일러 / 데이터 로드
# ─────────────────────────────────────────────────────────────
model = None
scaler = None
df = None

model_path = os.path.join(APP_DIR, "lung_model.pkl")
scaler_path = os.path.join(APP_DIR, "scaler.pkl")
df_path = os.path.join(APP_DIR, "lung.csv")


# 1단계: CSV 로드
try:
    df = load_csv_robust(df_path)

except Exception as e:

    st.error(
        f"lung.csv 로드 실패: {type(e).__name__}: {e}"
    )

    st.warning(
        f"`lung.csv` 파일이 "
        f"`{APP_DIR}` 폴더에 있는지 확인해주세요."
    )

    st.stop()


# 2단계: pkl 로드 시도
try:

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 실제 동작 확인
    _ = scaler.transform(
        df[["나이", "흡연량", "음주량"]].head(1)
    )

    _ = model.predict(
        scaler.transform(
            df[["나이", "흡연량", "음주량"]].head(1)
        )
    )

    st.success("저장된 모델(pkl) 로드 성공!")

except Exception as e:

    st.info(
        f"⚙️ 저장된 모델을 사용할 수 없어서 "
        f"`lung.csv` 로 모델을 다시 학습합니다.\n\n"
        f"(원인: `{type(e).__name__}` "
        f"— 보통 sklearn 버전 차이 때문입니다)"
    )

    try:

        model, scaler = rebuild_from_csv(df)

        X = df[["나이", "흡연량", "음주량"]]

        df["cluster"] = model.predict(
            scaler.transform(X)
        )

        st.success(
            "재학습 완료! "
            "(K-means, K=4, random_state=42)"
        )

    except Exception as e2:

        st.error(
            f"재학습도 실패: "
            f"{type(e2).__name__}: {e2}"
        )

        st.stop()


# ─────────────────────────────────────────────────────────────
# 사이드바 군집 정보
# ─────────────────────────────────────────────────────────────
cluster_info = build_cluster_info(df)

sidebar_lines = ["**군집 정보 (K=4)**"]

for c in sorted(cluster_info.keys()):

    sidebar_lines.append(
        f"- **{c}번**: {cluster_info[c]}"
    )

st.sidebar.markdown("\n".join(sidebar_lines))


# ─────────────────────────────────────────────────────────────
# 입력
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 환자 정보 입력</div>', unsafe_allow_html=True)

age = st.slider("나이", 10.0, 90.0, 40.0)

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
    columns=["나이", "흡연량", "음주량"]
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
            model.predict(new_patient_scaled)[0]
        )

        st.success(
            f"이 환자는 "
            f"**{pred_cluster}번 군집**에 속합니다."
        )

        st.info(
            f"군집 의미: "
            f"{cluster_info.get(pred_cluster, '알 수 없음')}"
        )

        # ─────────────────────────────────────────
        # 시각화 1 : 나이 vs 흡연량
        # ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f6faff")

        scatter = ax.scatter(
            df["나이"],
            df["흡연량"],
            c=df["cluster"],
            alpha=0.6,
            cmap="viridis",
            s=55,
            edgecolors="white",
            linewidths=0.6
        )

        ax.scatter(
            age,
            smoking,
            c="#e11d48",
            s=340,
            marker="X",
            edgecolors="white",
            linewidths=2.2,
            zorder=5
        )

        ax.set_xlabel("나이", fontsize=12, labelpad=8)
        ax.set_ylabel("흡연량", fontsize=12, labelpad=8)

        ax.set_title(
            f"폐암 환자 군집 분석 "
            f"(입력값은 {pred_cluster}번 군집)",
            fontsize=14, fontweight="bold", pad=14
        )

        ax.grid(True, linestyle="--", alpha=0.35)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        st.pyplot(fig)

        plt.close(fig)

        # ─────────────────────────────────────────
        # 시각화 2 : 나이 vs 음주량
        # ─────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        fig2.patch.set_facecolor("#ffffff")
        ax2.set_facecolor("#f6faff")

        ax2.scatter(
            df["나이"],
            df["음주량"],
            c=df["cluster"],
            alpha=0.6,
            cmap="viridis",
            s=55,
            edgecolors="white",
            linewidths=0.6
        )

        ax2.scatter(
            age,
            drinking,
            c="#e11d48",
            s=340,
            marker="X",
            edgecolors="white",
            linewidths=2.2,
            zorder=5
        )

        ax2.set_xlabel("나이", fontsize=12, labelpad=8)
        ax2.set_ylabel("음주량", fontsize=12, labelpad=8)

        ax2.set_title("나이 vs 음주량", fontsize=14, fontweight="bold", pad=14)

        ax2.grid(True, linestyle="--", alpha=0.35)
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)

        st.pyplot(fig2)

        plt.close(fig2)

    except Exception as e:

        st.error(
            f"군집 예측 실패: "
            f"{type(e).__name__}: {e}"
        )