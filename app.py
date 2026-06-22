import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# =========================================================
# 页面配置
# =========================================================
st.set_page_config(
    page_title="浙江就业市场智能洞察平台",
    layout="wide",
    page_icon="📊"
)

DATA_PATH = Path("data/clean_jobs.csv")
STREAM_SIMULATION_FACTOR = 5  # 原始数据基础上额外生成 5 个批次，形成“流数据”效果


# =========================================================
# 数据加载与基础处理
# =========================================================
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error("未找到 data/clean_jobs.csv，请先运行 preprocess\\clean_jobs.py 完成数据清洗。")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    required_cols = [
        "job_name", "company_name", "city",
        "salary_min", "salary_max", "salary_avg",
        "degree_clean", "degree_level",
        "salary_level", "job_direction",
        "skill_keywords", "company_property", "source"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df["salary_avg"] = pd.to_numeric(df["salary_avg"], errors="coerce")
    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")
    df = df.dropna(subset=["salary_avg"]).copy()

    degree_map = {
        "不限": 0,
        "专科及以上": 1,
        "大专": 1,
        "本科及以上": 2,
        "本科": 2,
        "硕士及以上": 3,
        "硕士": 3,
        "博士及以上": 4,
        "博士": 4,
    }

    if "degree_level" not in df.columns:
        df["degree_level"] = df["degree_clean"].map(degree_map).fillna(0)
    else:
        df["degree_level"] = pd.to_numeric(df["degree_level"], errors="coerce")
        df["degree_level"] = df["degree_level"].fillna(df["degree_clean"].map(degree_map)).fillna(0)

    if "salary_level" not in df.columns or df["salary_level"].astype(str).str.strip().eq("").all():
        df["salary_level"] = df["salary_avg"].apply(get_salary_level)

    # 将 500+ 静态样本扩展为多批次“流数据”样本，便于展示动态分析效果
    df = simulate_stream_data(df, factor=STREAM_SIMULATION_FACTOR)

    return df

def build_insights(df):
    """决策引擎：把数据变成结论"""

    if df.empty:
        return {}

    city_rank = df["city"].value_counts()
    direction_rank = df["job_direction"].value_counts()
    skill_rank = df["skill_keywords"].value_counts()

    avg_salary = df["salary_avg"].mean()

    top_city = city_rank.index[0]
    top_direction = direction_rank.index[0]

    high_salary_df = df[df["salary_avg"] >= 12000]
    high_salary_city = high_salary_df["city"].value_counts().index[0] if not high_salary_df.empty else "暂无"

    return {
        "top_city": top_city,
        "top_direction": top_direction,
        "avg_salary": avg_salary,
        "high_salary_city": high_salary_city,
        "city_rank": city_rank,
        "direction_rank": direction_rank,
        "skill_rank": skill_rank
    }

def get_salary_level(salary):
    if pd.isna(salary):
        return "薪资未知"
    salary = float(salary)
    if salary < 5000:
        return "5K以下"
    elif salary < 8000:
        return "5K-8K"
    elif salary < 12000:
        return "8K-12K"
    elif salary < 20000:
        return "12K-20K"
    else:
        return "20K以上"


def simulate_stream_data(df, factor=5, seed=42):
    """
    流数据模拟：在不改变原始岗位结构的前提下，生成多个“批次流入”的岗位样本。
    说明：这是可视化实验中的流式数据增强，用于展示实时增量分析效果。
    """
    if df.empty or factor <= 0:
        df = df.copy()
        df["stream_batch"] = 0
        df["stream_time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        df["data_type"] = "原始采集"
        return df

    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp.now().floor("min")

    frames = []
    base = df.copy()
    base["stream_batch"] = 0
    base["stream_time"] = base_time.strftime("%Y-%m-%d %H:%M")
    base["data_type"] = "原始采集"
    frames.append(base)

    for batch in range(1, factor + 1):
        part = df.copy()

        # 模拟流数据批次到达时薪资的轻微波动，避免简单复制导致过于死板
        noise = rng.normal(loc=1.0, scale=0.045, size=len(part))
        part["salary_avg"] = (part["salary_avg"] * noise).clip(lower=1000).round(0)

        if "salary_min" in part.columns:
            part["salary_min"] = pd.to_numeric(part["salary_min"], errors="coerce")
            part["salary_min"] = (part["salary_min"] * noise).clip(lower=0).round(0)

        if "salary_max" in part.columns:
            part["salary_max"] = pd.to_numeric(part["salary_max"], errors="coerce")
            part["salary_max"] = (part["salary_max"] * noise).clip(lower=0).round(0)

        part["salary_level"] = part["salary_avg"].apply(get_salary_level)
        part["stream_batch"] = batch
        part["stream_time"] = (base_time + pd.Timedelta(minutes=batch * 5)).strftime("%Y-%m-%d %H:%M")
        part["data_type"] = "模拟流入"
        frames.append(part)

    return pd.concat(frames, ignore_index=True)


def parse_search_keywords(keyword):
    """
    把用户输入拆成多个检索词，并加入常见同义词/近义词。
    这样输入“医生”时，也能命中“医师、临床、医学、护士、药师”等相关岗位。
    """
    keyword = str(keyword).strip()
    if not keyword:
        return []

    parts = re.split(r"[\s,，、/;；|]+", keyword)
    parts = [p.strip() for p in parts if p.strip()]

    # 保留完整短语，同时保留拆分词
    if keyword not in parts:
        parts.insert(0, keyword)

    synonym_map = {
        "医生": ["医生", "医师", "医士", "临床", "医学", "医疗", "卫生", "药师", "护士", "护理", "检验", "影像", "康复"],
        "医师": ["医生", "医师", "临床", "医学", "医疗"],
        "护士": ["护士", "护理", "医疗", "卫生"],
        "教师": ["教师", "老师", "教学", "教育", "培训", "讲师", "助教"],
        "老师": ["教师", "老师", "教学", "教育", "培训", "讲师", "助教"],
        "程序员": ["程序员", "开发", "软件", "Java", "Python", "前端", "后端", "工程师"],
        "会计": ["会计", "财务", "审计", "出纳", "税务"],
        "销售": ["销售", "市场", "客户", "商务", "业务"],
        "运营": ["运营", "新媒体", "直播", "电商", "内容", "用户"],
        "设计": ["设计", "CAD", "制图", "美工", "视觉"],
        "机械": ["机械", "工程", "设备", "制造", "自动化", "机电"],
        "数据分析": ["数据分析", "数据", "Excel", "Python", "SQL", "统计", "分析"],
    }

    expanded = []
    for item in parts:
        expanded.append(item)
        if item in synonym_map:
            expanded.extend(synonym_map[item])

    seen = set()
    result = []
    for p in expanded:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def weighted_keyword_search(df, keyword):
    """
    多字段加权检索：岗位名 > 技能关键词/岗位方向 > 公司名。
    返回按 search_score 降序排列的数据。
    """
    keywords = parse_search_keywords(keyword)
    if not keywords or df.empty:
        return df.copy()

    result = df.copy()
    result["search_score"] = 0

    weighted_fields = {
        "job_name": 5,
        "skill_keywords": 4,
        "job_direction": 3,
        "company_name": 2,
        "city": 1,
        "degree_clean": 1,
    }

    for kw in keywords:
        pattern = re.escape(kw)
        for col, weight in weighted_fields.items():
            if col in result.columns:
                hit = result[col].astype(str).str.contains(pattern, case=False, na=False, regex=True)
                result.loc[hit, "search_score"] += weight

    result = result[result["search_score"] > 0].copy()
    if result.empty:
        return result

    sort_cols = ["search_score"]
    ascending = [False]
    if "salary_avg" in result.columns:
        sort_cols.append("salary_avg")
        ascending.append(False)

    return result.sort_values(sort_cols, ascending=ascending)


# =========================================================
# CSS：答辩级 UI 风格
# =========================================================
def add_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1e3a8a;
            --cyan: #38bdf8;
            --bg: #f5f7fb;
            --card: rgba(255,255,255,0.96);
            --text: #0f172a;
            --muted: #64748b;
            --border: rgba(148, 163, 184, 0.22);
        }

        html, body {
            margin: 0 !important;
            padding: 0 !important;
        }

        .stApp {
            background:
                radial-gradient(circle at 22% 8%, rgba(37, 99, 235, 0.08), transparent 26%),
                linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
        }

        /* 顶部系统栏透明，减少上方空白感 */
        [data-testid="stHeader"] {
            background: rgba(255,255,255,0) !important;
            height: 2.2rem;
        }

        /* 主体真正铺开：解决左右空白 */
        .block-container {
            max-width: 100% !important;
            width: 100% !important;
            padding-top: 0.35rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-bottom: 2rem !important;
        }

        /* 让内部块不再被二次压窄 */
        [data-testid="stVerticalBlock"] {
            gap: 0.8rem;
        }

        /* 侧边栏 */
        section[data-testid="stSidebar"] {
            width: 285px !important;
            min-width: 285px !important;
            background: linear-gradient(180deg, #eef4ff 0%, #e8eef8 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.25);
        }

        section[data-testid="stSidebar"] > div {
            width: 285px !important;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #0f172a;
        }

        section[data-testid="stSidebar"] label {
            font-size: 15px !important;
            font-weight: 700 !important;
            color: #0f172a !important;
        }


        /* 侧边栏顶部操作按钮 */
        section[data-testid="stSidebar"] .stButton > button {
            border-radius: 12px !important;
            min-height: 42px !important;
            font-size: 15px !important;
            font-weight: 850 !important;
            border: 1px solid rgba(37,99,235,0.20) !important;
            box-shadow: 0 6px 16px rgba(15,23,42,0.06) !important;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(37,99,235,0.45) !important;
        }

        /* 多选标签 */
        span[data-baseweb="tag"] {
            background-color: #dbeafe !important;
            color: #1d4ed8 !important;
            border: 1px solid #bfdbfe !important;
            border-radius: 999px !important;
            font-weight: 700 !important;
            font-size: 14px !important;
        }

        div[data-baseweb="select"] > div {
            border-radius: 12px !important;
            border-color: rgba(148,163,184,0.35) !important;
            background-color: rgba(255,255,255,0.92) !important;
        }

        /* 顶部长方形标题区：铺满主内容，不留左右空白 */
        .hero-card {
            width: 100% !important;
            box-sizing: border-box !important;
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(15, 76, 129, 0.98) 0%, rgba(37, 99, 235, 0.98) 55%, rgba(56, 189, 248, 0.95) 100%);
            color: white;
            padding: 24px 30px;
            border-radius: 0px;
            margin: 0 0 14px 0;
            box-shadow: 0 14px 36px rgba(37, 99, 235, 0.22);
        }

        .hero-card::after {
            content: "";
            position: absolute;
            right: -70px;
            top: -80px;
            width: 220px;
            height: 220px;
            background: rgba(255,255,255,0.14);
            border-radius: 50%;
        }

        .hero-eyebrow {
            font-size: 13px;
            letter-spacing: 1.8px;
            text-transform: uppercase;
            opacity: 0.88;
            margin-bottom: 8px;
            font-weight: 800;
        }

        .hero-title {
            font-size: 34px;
            font-weight: 950;
            margin-bottom: 10px;
            line-height: 1.22;
        }

        .hero-subtitle {
            font-size: 16px;
            opacity: 0.96;
            line-height: 1.8;
            max-width: 1200px;
            font-weight: 600;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }

        .hero-badge {
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.24);
            font-size: 13px;
            font-weight: 800;
        }

        /* Tab 导航栏：整体铺满，字体调大 */
        div[data-testid="stTabs"] {
            background: rgba(255,255,255,0.92);
            padding: 8px 12px 0 12px;
            border-radius: 0px;
            border: 1px solid rgba(148,163,184,0.22);
            box-shadow: 0 6px 18px rgba(15,23,42,0.04);
            margin-top: 0 !important;
            margin-bottom: 12px !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }

        div[data-testid="stTabs"] [role="tablist"] {
            gap: 18px;
        }

        div[data-testid="stTabs"] button {
            padding: 11px 18px !important;
            border-radius: 10px 10px 0 0 !important;
        }

        div[data-testid="stTabs"] button p {
            font-size: 17px !important;
            font-weight: 850 !important;
            color: #1e293b !important;
            line-height: 1.3 !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: #2563eb !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] p {
            color: white !important;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: #ef4444 !important;
            height: 3px !important;
        }

        /* 标题层级 */
        h1 {
            font-size: 34px !important;
            font-weight: 900 !important;
        }

        h2 {
            font-size: 28px !important;
            font-weight: 900 !important;
            margin-top: 12px !important;
        }

        h3 {
            font-size: 22px !important;
            font-weight: 850 !important;
        }

        /* 指标卡片 */
        .metric-card {
            background: var(--card);
            backdrop-filter: blur(14px);
            padding: 16px 18px;
            border-radius: 18px;
            border: 1px solid var(--border);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
            margin-bottom: 14px;
            min-height: 112px;
            transition: 0.25s ease;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.10);
        }

        .metric-title {
            font-size: 14px;
            color: #64748b;
            margin-bottom: 6px;
            font-weight: 800;
        }

        .metric-value {
            font-size: 30px;
            font-weight: 950;
            color: #0f172a;
            margin-bottom: 4px;
            letter-spacing: -0.5px;
        }

        .metric-note {
            font-size: 13px;
            color: #94a3b8;
            font-weight: 700;
        }

        .section-intro {
            background: rgba(255,255,255,0.95);
            padding: 14px 18px;
            border-radius: 15px;
            border: 1px solid var(--border);
            margin: 12px 0 16px 0;
            color: #475569;
            line-height: 1.75;
            box-shadow: 0 6px 18px rgba(15,23,42,0.045);
            font-size: 15px;
        }

        .insight-box {
            background: linear-gradient(90deg, #eff6ff 0%, #f8fbff 100%);
            border-left: 5px solid #2563eb;
            padding: 15px 18px;
            border-radius: 14px;
            color: #1e293b;
            margin-top: 14px;
            margin-bottom: 16px;
            line-height: 1.8;
            font-size: 15px;
            box-shadow: 0 6px 18px rgba(37,99,235,0.07);
        }

        .chart-title {
            font-size: 22px;
            font-weight: 900;
            color: #0f172a;
            margin: 8px 0 10px 0;
        }

        .subtle-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(148,163,184,0.4), transparent);
            margin: 16px 0;
        }

        .stDataFrame {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.22);
            box-shadow: 0 6px 18px rgba(15,23,42,0.05);
        }

        .js-plotly-plot {
            border-radius: 16px;
        }


        /* ===== 封面入口页：全屏官网式 Hero ===== */
        .cover-shell {
            min-height: calc(100vh - 2.2rem);
            width: 100%;
            position: relative;
            overflow: hidden;
            border-radius: 0;
            margin: -0.35rem -0.75rem 0 -0.75rem;
            color: white;
            background:
                linear-gradient(90deg, rgba(8, 26, 67, 0.78) 0%, rgba(12, 52, 120, 0.66) 36%, rgba(37, 99, 235, 0.30) 100%),
                radial-gradient(circle at 78% 22%, rgba(56,189,248,0.34), transparent 24%),
                linear-gradient(135deg, #0f172a 0%, #1e3a8a 48%, #38bdf8 100%);
            box-shadow: inset 0 -180px 180px rgba(15, 23, 42, 0.18);
        }

        .cover-shell::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px),
                linear-gradient(180deg, rgba(255,255,255,0.08) 1px, transparent 1px);
            background-size: 72px 72px;
            opacity: 0.22;
        }

        .cover-shell::after {
            content: "";
            position: absolute;
            right: -12vw;
            bottom: -22vh;
            width: 58vw;
            height: 58vw;
            background: rgba(255,255,255,0.09);
            transform: rotate(38deg);
            border-radius: 48px;
        }

        .cover-topbar {
            position: relative;
            z-index: 2;
            height: 68px;
            padding: 0 52px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 23, 42, 0.20);
            border-bottom: 1px solid rgba(255,255,255,0.16);
            backdrop-filter: blur(10px);
        }

        .cover-logo {
            display: flex;
            align-items: center;
            gap: 14px;
            font-weight: 950;
            letter-spacing: 0.8px;
            font-size: 19px;
        }

        .cover-logo-mark {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: rgba(255,255,255,0.18);
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(255,255,255,0.26);
            box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        }

        .cover-nav {
            display: flex;
            gap: 34px;
            align-items: center;
            font-size: 15px;
            font-weight: 800;
            opacity: 0.95;
        }

        .cover-main {
            position: relative;
            z-index: 2;
            min-height: calc(100vh - 70px);
            display: grid;
            grid-template-columns: 1.08fr 0.92fr;
            align-items: center;
            gap: 46px;
            padding: 44px 62px 76px 62px;
        }

        .cover-eyebrow {
            font-size: 14px;
            letter-spacing: 3px;
            font-weight: 950;
            opacity: 0.86;
            margin-bottom: 18px;
            text-transform: uppercase;
        }

        .cover-title {
            font-size: clamp(46px, 5vw, 78px);
            line-height: 1.04;
            font-weight: 950;
            letter-spacing: -2.2px;
            margin-bottom: 24px;
            text-shadow: 0 14px 34px rgba(0,0,0,0.22);
        }

        .cover-subtitle {
            max-width: 900px;
            font-size: 18px;
            line-height: 2;
            font-weight: 700;
            opacity: 0.94;
            margin-bottom: 28px;
        }

        .cover-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 30px;
        }

        .cover-tag {
            padding: 10px 16px;
            border-radius: 999px;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.24);
            font-size: 14px;
            font-weight: 900;
            box-shadow: 0 8px 24px rgba(0,0,0,0.10);
        }

        .cover-start-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            height: 52px;
            min-width: 220px;
            padding: 0 26px;
            background: #ef4444;
            color: white !important;
            border-radius: 999px;
            text-decoration: none !important;
            font-size: 16px;
            font-weight: 950;
            box-shadow: 0 18px 40px rgba(239,68,68,0.32);
            transition: 0.22s ease;
        }

        .cover-start-link:hover {
            transform: translateY(-2px);
            background: #dc2626;
            color: white !important;
        }

        .cover-panel {
            justify-self: end;
            width: min(520px, 100%);
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(255,255,255,0.23);
            border-radius: 28px;
            padding: 26px;
            backdrop-filter: blur(16px);
            box-shadow: 0 28px 80px rgba(0,0,0,0.18);
        }

        .cover-panel-title {
            font-size: 20px;
            font-weight: 950;
            margin-bottom: 16px;
        }

        .cover-metrics {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin-top: 12px;
        }

        .cover-metric {
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 18px;
            padding: 18px 18px;
        }

        .cover-metric-label {
            font-size: 13px;
            opacity: 0.82;
            font-weight: 900;
            margin-bottom: 8px;
        }

        .cover-metric-value {
            font-size: 32px;
            font-weight: 950;
            letter-spacing: -0.8px;
        }

        .cover-note {
            margin-top: 18px;
            font-size: 14px;
            opacity: 0.92;
            line-height: 1.9;
            padding: 14px 16px;
            border-radius: 16px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.14);
        }

        .cover-bottom-hint {
            position: absolute;
            z-index: 3;
            left: 50%;
            bottom: 24px;
            transform: translateX(-50%);
            font-size: 28px;
            opacity: 0.88;
            animation: coverFloat 1.8s ease-in-out infinite;
        }

        @keyframes coverFloat {
            0%, 100% { transform: translateX(-50%) translateY(0); }
            50% { transform: translateX(-50%) translateY(8px); }
        }

        @media (max-width: 980px) {
            .cover-main {
                grid-template-columns: 1fr;
                padding: 34px 28px 70px 28px;
            }
            .cover-panel {
                justify-self: stretch;
            }
            .cover-nav {
                display: none;
            }
            .cover-title {
                font-size: 42px;
            }
        }

        .stButton > button[kind="primary"] {
            border-radius: 999px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 900 !important;
            font-size: 16px !important;
        }



        /* ===== 准实时/流数据状态条 ===== */
        .live-status-bar {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 12px 0 16px 0;
        }
        .live-status-card {
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 16px;
            padding: 13px 15px;
            box-shadow: 0 8px 22px rgba(15,23,42,0.06);
        }
        .live-status-label {
            color: #64748b;
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 5px;
        }
        .live-status-value {
            color: #0f172a;
            font-size: 20px;
            font-weight: 950;
        }
        .live-dot {
            display: inline-block;
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #22c55e;
            margin-right: 7px;
            box-shadow: 0 0 0 6px rgba(34,197,94,0.12);
            vertical-align: middle;
        }
        .realtime-note {
            background: linear-gradient(90deg, #f0fdf4 0%, #eff6ff 100%);
            border-left: 5px solid #22c55e;
            border-radius: 14px;
            padding: 12px 16px;
            margin: 4px 0 14px 0;
            color: #1e293b;
            font-size: 14px;
            line-height: 1.75;
            box-shadow: 0 6px 18px rgba(34,197,94,0.06);
        }
        @media (max-width: 1100px) {
            .live-status-bar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }



        /* ===== 侧边栏检索区优化 ===== */
        section[data-testid="stSidebar"] label {
            font-size: 15px !important;
            font-weight: 850 !important;
            color: #0f172a !important;
        }

        section[data-testid="stSidebar"] .stButton > button {
            min-height: 44px !important;
            border-radius: 14px !important;
            font-size: 15px !important;
            font-weight: 850 !important;
            border: 1px solid rgba(148,163,184,0.32) !important;
            box-shadow: 0 6px 16px rgba(15,23,42,0.05) !important;
        }

        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #64748b !important;
            font-size: 13px !important;
            line-height: 1.65 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def metric_card(title, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def insight_box(text):
    st.markdown(
        f"""
        <div class="insight-box">
        <b>自动洞察：</b>{text}
        </div>
        """,
        unsafe_allow_html=True
    )


def section_intro(text):
    st.markdown(
        f"""
        <div class="section-intro">
        {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def chart_title(text):
    st.markdown(f'<div class="chart-title">{text}</div>', unsafe_allow_html=True)



def get_stream_status(df):
    """返回当前数据的流数据状态，用于在页面上明确展示“准实时/模拟流入”。"""
    if df is None or df.empty:
        return {"raw_count": 0, "stream_count": 0, "batch_count": 0, "latest_time": "暂无"}

    if "data_type" in df.columns:
        raw_count = int((df["data_type"].astype(str) == "原始采集").sum())
        stream_count = int((df["data_type"].astype(str) == "模拟流入").sum())
    else:
        raw_count = len(df)
        stream_count = 0

    if "stream_batch" in df.columns:
        batch_count = int(pd.to_numeric(df["stream_batch"], errors="coerce").fillna(0).max()) + 1
    else:
        batch_count = 1

    if "stream_time" in df.columns and not df["stream_time"].dropna().empty:
        latest_time = str(df["stream_time"].dropna().iloc[-1])
    else:
        latest_time = "暂无"

    return {"raw_count": raw_count, "stream_count": stream_count, "batch_count": batch_count, "latest_time": latest_time}


def show_stream_status(df):
    """在首页仪表盘明确展示数据更新状态，避免看起来像静态 CSV。"""
    status = get_stream_status(df)
    st.markdown(
        f'''
        <div class="live-status-bar">
            <div class="live-status-card">
                <div class="live-status-label"><span class="live-dot"></span>数据状态</div>
                <div class="live-status-value">准实时模拟</div>
            </div>
            <div class="live-status-card">
                <div class="live-status-label">原始采集岗位</div>
                <div class="live-status-value">{status["raw_count"]:,}</div>
            </div>
            <div class="live-status-card">
                <div class="live-status-label">模拟流入岗位</div>
                <div class="live-status-value">{status["stream_count"]:,}</div>
            </div>
            <div class="live-status-card">
                <div class="live-status-label">最新批次时间</div>
                <div class="live-status-value">{status["latest_time"]}</div>
            </div>
        </div>
        <div class="realtime-note">
            <b>数据更新说明：</b>当前版本采用“原始岗位数据 + 多批次模拟流入”的准实时展示方式，
            页面会显示批次、时间和流入规模；点击左侧“刷新本地数据”可重新读取最新 CSV。
            如果接入定时爬虫，可将这里升级为真正的在线实时更新。
        </div>
        ''',
        unsafe_allow_html=True
    )


# =========================================================
# 图表统一风格
# =========================================================
COLOR_SCALE_BLUE = ["#e0f2fe", "#7dd3fc", "#38bdf8", "#2563eb", "#1e3a8a"]
COLOR_SCALE_TEAL = ["#ccfbf1", "#5eead4", "#14b8a6", "#0f766e"]
COLOR_SCALE_ORANGE = ["#ffedd5", "#fdba74", "#f97316", "#c2410c"]
COLOR_SCALE_PURPLE = ["#ede9fe", "#c4b5fd", "#8b5cf6", "#5b21b6"]


def style_fig(fig, height=460, showlegend=True):
    fig.update_layout(
        height=height,
        template="simple_white",
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(size=12, color="#334155"),
        title=dict(text=""),   # 关键：避免出现 undefined
        margin=dict(l=28, r=28, t=20, b=36),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=11)
        ),
        showlegend=showlegend
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False,
        linecolor="rgba(148,163,184,0.25)",
        title_font=dict(size=12),
        tickfont=dict(size=11)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False,
        linecolor="rgba(148,163,184,0.25)",
        title_font=dict(size=12),
        tickfont=dict(size=11)
    )
    return fig



# =========================================================
# 浙江省城市空间分布图
# =========================================================
ZHEJIANG_CITY_COORDS = {
    "杭州": (30.2741, 120.1551),
    "宁波": (29.8683, 121.5440),
    "温州": (27.9938, 120.6994),
    "嘉兴": (30.7461, 120.7555),
    "湖州": (30.8943, 120.0868),
    "绍兴": (30.0303, 120.5802),
    "金华": (29.0791, 119.6474),
    "衢州": (28.9701, 118.8593),
    "舟山": (30.0160, 122.1069),
    "台州": (28.6564, 121.4208),
    "丽水": (28.4676, 119.9229),
}


def build_city_geo_df(df):
    """按城市聚合岗位数量、企业数量和平均薪资，并补充经纬度。"""
    if df.empty or "city" not in df.columns:
        return pd.DataFrame(columns=["城市", "岗位数量", "覆盖企业", "平均薪资", "lat", "lon"])

    city_geo = (
        df.groupby("city")
        .agg(
            岗位数量=("job_name", "count"),
            覆盖企业=("company_name", "nunique"),
            平均薪资=("salary_avg", "mean")
        )
        .reset_index()
        .rename(columns={"city": "城市"})
    )

    city_geo["城市"] = city_geo["城市"].astype(str).str.replace("市", "", regex=False).str.strip()
    city_geo["lat"] = city_geo["城市"].map(lambda x: ZHEJIANG_CITY_COORDS.get(x, (None, None))[0])
    city_geo["lon"] = city_geo["城市"].map(lambda x: ZHEJIANG_CITY_COORDS.get(x, (None, None))[1])
    city_geo = city_geo.dropna(subset=["lat", "lon"]).copy()
    city_geo["平均薪资"] = city_geo["平均薪资"].round(0)
    city_geo = city_geo.sort_values("岗位数量", ascending=False)
    return city_geo


def draw_city_map(df):
    """浙江省就业岗位空间分布图：气泡大小代表岗位数量，颜色代表平均薪资。"""
    city_geo = build_city_geo_df(df)
    if city_geo.empty:
        return None

    fig = px.scatter_mapbox(
        city_geo,
        lat="lat",
        lon="lon",
        size="岗位数量",
        color="平均薪资",
        hover_name="城市",
        hover_data={
            "岗位数量": True,
            "覆盖企业": True,
            "平均薪资": ":,.0f",
            "lat": False,
            "lon": False,
        },
        size_max=48,
        zoom=6.25,
        center={"lat": 29.4, "lon": 120.55},
        mapbox_style="open-street-map",
        color_continuous_scale=COLOR_SCALE_BLUE,
    )
    fig.update_traces(marker=dict(opacity=0.78))
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        coloraxis_colorbar=dict(title="平均薪资"),
        font=dict(size=12, color="#334155"),
    )
    return fig


# =========================================================
# 技能处理
# =========================================================
def split_skills(skill_text):
    if pd.isna(skill_text) or str(skill_text).strip() in ["", "None", "nan"]:
        return []

    text = str(skill_text).replace("，", "、").replace(",", "、").replace("/", "、")
    return [x.strip() for x in text.split("、") if x.strip()]


def get_skill_frequency(df):
    skills = []
    for text in df["skill_keywords"].fillna(""):
        skills.extend(split_skills(text))

    if not skills:
        return pd.DataFrame(columns=["技能关键词", "出现次数"])

    skill_df = pd.Series(skills).value_counts().reset_index()
    skill_df.columns = ["技能关键词", "出现次数"]
    return skill_df


def make_skill_matrix(skill_freq, top_n=36):
    """
    技能关键词矩阵图：替代传统词云。
    优点：不依赖中文字体文件，不会出现方框；同时比气泡云更紧凑，适合报告截图。
    """
    if skill_freq.empty:
        return None

    top = skill_freq.head(top_n).copy().reset_index(drop=True)
    cols = 6
    rows = int(np.ceil(len(top) / cols))

    labels = np.full((rows, cols), "", dtype=object)
    values = np.full((rows, cols), np.nan, dtype=float)

    for i, row in top.iterrows():
        r = i // cols
        c = i % cols
        labels[r, c] = f"{row['技能关键词']}<br>{int(row['出现次数'])}"
        values[r, c] = row["出现次数"]

    fig = px.imshow(
        values,
        text_auto=False,
        color_continuous_scale=COLOR_SCALE_BLUE,
        aspect="auto",
    )
    fig.update_traces(
        text=labels,
        texttemplate="%{text}",
        hovertemplate="技能关键词：%{text}<extra></extra>",
        textfont=dict(size=16, color="#0f172a", family="Microsoft YaHei, SimHei, Arial")
    )
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    return fig


def skill_to_category(skill):
    s = str(skill).lower()
    if any(k.lower() in s for k in ["python", "java", "数据", "excel", "sql", "开发", "算法", "分析"]):
        return "技术/数据能力"
    if any(k in str(skill) for k in ["机械", "设计", "cad", "工程", "制造", "plc", "自动化", "电气"]):
        return "制造/工程能力"
    if any(k in str(skill) for k in ["销售", "市场", "商务", "客户", "客服"]):
        return "销售/服务能力"
    if any(k in str(skill) for k in ["运营", "新媒体", "直播", "电商", "短视频", "剪辑", "内容"]):
        return "运营/传播能力"
    if any(k in str(skill) for k in ["教师", "教学", "教育", "培训", "课程"]):
        return "教育/培训能力"
    if any(k in str(skill) for k in ["英语", "外贸", "跨境", "外语"]):
        return "外贸/语言能力"
    if any(k in str(skill) for k in ["财务", "会计", "审计", "行政", "人力"]):
        return "职能管理能力"
    return "通用综合能力"


def make_skill_category_chart(skill_freq):
    """把零散技能词汇归并为能力类别，形成更有分析价值的能力结构图。"""
    if skill_freq.empty:
        return None

    cat_df = skill_freq.copy()
    cat_df["能力类别"] = cat_df["技能关键词"].apply(skill_to_category)
    cat_sum = cat_df.groupby("能力类别", as_index=False)["出现次数"].sum().sort_values("出现次数", ascending=True)

    fig = px.bar(
        cat_sum,
        x="出现次数",
        y="能力类别",
        orientation="h",
        text="出现次数",
        color="出现次数",
        color_continuous_scale=COLOR_SCALE_TEAL,
        template="simple_white"
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False)
    return style_fig(fig, 480, showlegend=False)


# =========================================================
# 聚类处理
# =========================================================
# =========================================================
# 聚类处理
# =========================================================
def build_cluster_data(df):
    cluster_df = df.copy()

    cluster_df["skill_count"] = cluster_df["skill_keywords"].fillna("").apply(
        lambda x: len(split_skills(x))
    )

    cluster_df["degree_level"] = pd.to_numeric(cluster_df["degree_level"], errors="coerce").fillna(0)
    cluster_df["salary_avg"] = pd.to_numeric(cluster_df["salary_avg"], errors="coerce").fillna(0)
    cluster_df["salary_min"] = pd.to_numeric(cluster_df.get("salary_min", 0), errors="coerce").fillna(cluster_df["salary_avg"])
    cluster_df["salary_max"] = pd.to_numeric(cluster_df.get("salary_max", 0), errors="coerce").fillna(cluster_df["salary_avg"])
    cluster_df["salary_range"] = (cluster_df["salary_max"] - cluster_df["salary_min"]).clip(lower=0)
    cluster_df["is_high_salary"] = (cluster_df["salary_avg"] >= 12000).astype(int)

    # 方向、城市、企业性质都纳入聚类，让岗位画像更丰富，不再只按学历和薪资挤在几条竖线上
    direction_dummies = pd.get_dummies(cluster_df["job_direction"], prefix="方向")
    city_dummies = pd.get_dummies(cluster_df["city"], prefix="城市")
    property_dummies = pd.get_dummies(cluster_df["company_property"], prefix="企业")

    # 城市和企业性质类别较多，保留出现较多的列，避免特征过散
    city_dummies = city_dummies.loc[:, city_dummies.sum().sort_values(ascending=False).head(8).index]
    property_dummies = property_dummies.loc[:, property_dummies.sum().sort_values(ascending=False).head(6).index]

    numeric_features = cluster_df[
        ["salary_avg", "salary_min", "salary_max", "salary_range", "degree_level", "skill_count", "is_high_salary"]
    ].fillna(0)

    features = pd.concat(
        [numeric_features, direction_dummies, city_dummies, property_dummies],
        axis=1
    )

    return cluster_df, features


def run_kmeans(df, n_clusters=4):
    cluster_df, features = build_cluster_data(df)

    if len(cluster_df) < n_clusters:
        cluster_df["cluster"] = 0
        cluster_df["聚类横轴"] = 0
        cluster_df["聚类纵轴"] = 0
        return cluster_df, None

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_df["cluster"] = model.fit_predict(x_scaled)

    # 使用 PCA 将多维岗位特征压缩到二维，用于绘制更丰富的聚类散点图
    if x_scaled.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(x_scaled)
        cluster_df["聚类横轴"] = coords[:, 0]
        cluster_df["聚类纵轴"] = coords[:, 1]
    else:
        cluster_df["聚类横轴"] = cluster_df["salary_avg"]
        cluster_df["聚类纵轴"] = cluster_df["skill_count"]

    return cluster_df, model


def get_cluster_name(row):
    direction = str(row["主要岗位方向"])
    avg_salary = row["平均薪资"]
    degree = str(row["主要学历要求"])

    if avg_salary >= 12000 and ("技术" in direction or "数据" in direction):
        return "高薪技术数据型岗位"
    if "制造" in direction or "工程" in direction:
        return "制造工程技能型岗位"
    if "销售" in direction or "市场" in direction:
        return "销售市场拓展型岗位"
    if "运营" in direction or "传媒" in direction:
        return "运营传媒增长型岗位"
    if "教育" in direction or "培训" in direction:
        return "教育培训服务型岗位"
    if "本科" in degree or "硕士" in degree or "博士" in degree:
        return "学历门槛提升型岗位"

    return "综合基础就业型岗位"


def make_cluster_display_names(cluster_summary):
    """
    给每个聚类生成唯一的岗位画像名称。
    之前多个 cluster 会被命名成同一个“制造工程技能型岗位”，图例只剩 3 类，视觉上不丰富。
    这里保留业务含义，同时加上主要城市/方向/序号，保证每个聚类都能单独展示。
    """
    names = []
    used = set()
    for _, row in cluster_summary.iterrows():
        base = get_cluster_name(row)
        city = str(row.get("主要城市", "")).strip()
        direction = str(row.get("主要岗位方向", "")).strip()
        cluster_no = int(row.get("cluster", 0)) + 1

        if city and direction:
            name = f"画像{cluster_no}｜{city}·{direction}"
        elif direction:
            name = f"画像{cluster_no}｜{direction}"
        else:
            name = f"画像{cluster_no}｜{base}"

        if name in used:
            name = f"画像{cluster_no}｜{base}"
        used.add(name)
        names.append(name)
    return names


# =========================================================
# 求职建议
# =========================================================
def generate_job_advice(skills, matched_df):
    if matched_df.empty:
        return "当前条件下暂无匹配岗位，建议放宽城市、薪资或岗位方向限制。"

    avg_salary = matched_df["salary_avg"].mean()
    top_city = matched_df["city"].value_counts().idxmax()
    top_direction = matched_df["job_direction"].value_counts().idxmax()

    advice = (
        f"根据当前条件，共匹配到 <b>{len(matched_df)}</b> 个岗位。"
        f"匹配岗位主要集中在 <b>{top_city}</b>，以 <b>{top_direction}</b> 方向为主，"
        f"平均薪资约为 <b>{avg_salary:,.0f}</b> 元/月。"
    )

    if skills.strip():
        advice += " 你输入的技能关键词已参与岗位名称、技能字段和岗位方向匹配，可进一步根据岗位表中的具体要求优化简历关键词。"
    else:
        advice += " 建议补充个人技能关键词，例如 Python、Java、Excel、销售、运营、外贸等，以提高岗位匹配精度。"

    if avg_salary >= 12000:
        advice += " 当前匹配岗位薪资水平较高，通常也意味着岗位门槛更高，建议重点关注学历、经验和技能要求。"
    elif avg_salary < 7000:
        advice += " 当前匹配结果以基础岗位为主，适合作为实习、应届或入门岗位参考。"
    else:
        advice += " 当前匹配岗位薪资处于中等区间，适合多数应届毕业生作为求职参考。"

    return advice



# =========================================================
# 封面入口页：不做复杂登录，只做系统入口与项目说明
# 说明：这里不要在 HTML 字符串里留空行，否则 Streamlit 的 markdown 会把后半段当普通文本显示。
# =========================================================
def show_cover_page(df):
    raw_count = len(df[df.get("data_type", "").astype(str).eq("原始采集")]) if "data_type" in df.columns else len(df)
    total_count = len(df)
    city_count = df["city"].nunique() if "city" in df.columns else 0
    company_count = df["company_name"].nunique() if "company_name" in df.columns else 0
    avg_salary = pd.to_numeric(df["salary_avg"], errors="coerce").mean() if "salary_avg" in df.columns else 0
    stream_status = get_stream_status(df)
    latest_time = stream_status["latest_time"]

    cover_html = f"""
<div class="cover-shell"><div class="cover-topbar"><div class="cover-logo"><div class="cover-logo-mark">📊</div><div>浙江就业市场智能洞察平台</div></div><div class="cover-nav"><span>数据总览</span><span>薪资学历</span><span>技能画像</span><span>岗位聚类</span><span>求职匹配</span></div></div><div class="cover-main"><div class="cover-left"><div class="cover-eyebrow">Zhejiang Employment Intelligence Platform</div><div class="cover-title">“数”说浙江就业<br>就业市场智能洞察平台</div><div class="cover-subtitle">面向大学生求职场景，基于国家大学生就业服务平台岗位数据，围绕城市分布、薪资结构、学历要求、技能关键词、岗位画像和求职匹配进行交互式分析。系统支持模拟流数据增强与多关键词加权检索，并在页面中显示最新批次时间、流入规模和数据更新状态，便于展示接近实时监测的分析过程。</div><div class="cover-tags"><span class="cover-tag">真实岗位数据</span><span class="cover-tag">模拟流数据</span><span class="cover-tag">加权岗位搜索</span><span class="cover-tag">K-Means岗位画像</span><span class="cover-tag">求职匹配建议</span></div><a class="cover-start-link" href="?enter=1" target="_self">进入分析驾驶舱 →</a></div><div class="cover-panel"><div class="cover-panel-title">准实时数据概览</div><div class="cover-metrics"><div class="cover-metric"><div class="cover-metric-label">原始采集岗位</div><div class="cover-metric-value">{raw_count:,}</div></div><div class="cover-metric"><div class="cover-metric-label">流数据增强后</div><div class="cover-metric-value">{total_count:,}</div></div><div class="cover-metric"><div class="cover-metric-label">覆盖城市</div><div class="cover-metric-value">{city_count}</div></div><div class="cover-metric"><div class="cover-metric-label">覆盖企业</div><div class="cover-metric-value">{company_count:,}</div></div><div class="cover-metric"><div class="cover-metric-label">平均薪资</div><div class="cover-metric-value">{avg_salary:,.0f}</div></div><div class="cover-metric"><div class="cover-metric-label">最新批次</div><div class="cover-metric-value">{latest_time}</div></div></div><div class="cover-note">当前为“原始采集数据 + 模拟流入批次”的准实时展示模式。进入系统后，左侧筛选项默认不选中，表示分析全部数据；选择城市、岗位方向、学历或输入关键词后，系统会自动刷新分析结果。</div></div></div><div class="cover-bottom-hint">⌄</div></div>
"""
    st.markdown(cover_html, unsafe_allow_html=True)


# =========================================================
# 主程序
# =========================================================
def main():
    add_custom_css()
    df = load_data()

    if "entered_dashboard" not in st.session_state:
        st.session_state["entered_dashboard"] = False

    # 封面页中的 HTML 按钮通过 ?enter=1 进入系统
    try:
        if st.query_params.get("enter") == "1":
            st.session_state["entered_dashboard"] = True
    except Exception:
        pass

    if not st.session_state["entered_dashboard"]:
        show_cover_page(df)
        return

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Zhejiang Employment Intelligence Platform</div>
            <div class="hero-title">“数”说浙江就业：就业市场智能洞察平台</div>
            <div class="hero-subtitle">
                基于国家大学生就业服务平台岗位数据，围绕浙江省城市分布、薪资水平、学历要求、技能关键词和岗位画像进行多维可视化分析，
                构建面向大学生求职决策的就业市场智能洞察平台。
            </div>
            <div class="badge-row">
                <span class="hero-badge">真实岗位数据</span>
                <span class="hero-badge">多维可视化</span>
                <span class="hero-badge">K-Means岗位画像</span>
                <span class="hero-badge">求职匹配建议</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    show_stream_status(df)

    # =====================================================
    # 侧边栏：岗位检索与筛选
    # =====================================================
    salary_min = int(df["salary_avg"].min())
    salary_max = int(df["salary_avg"].max())
    city_options = sorted(df["city"].dropna().unique().tolist())
    direction_options = sorted(df["job_direction"].dropna().unique().tolist())
    degree_options = sorted(df["degree_clean"].dropna().unique().tolist())

    def reset_to_latest_market():
        st.session_state["filter_keyword"] = ""
        st.session_state["filter_cities"] = city_options
        st.session_state["filter_directions"] = direction_options
        st.session_state["filter_degrees"] = degree_options
        st.session_state["filter_salary"] = (salary_min, salary_max)

    def back_to_cover():
        st.session_state["entered_dashboard"] = False
        try:
            st.query_params.clear()
        except Exception:
            try:
                st.experimental_set_query_params()
            except Exception:
                pass

    st.sidebar.markdown(
        """
        <div style="margin-top: 4px; margin-bottom: 18px;">
            <div style="font-size: 34px; font-weight: 950; line-height: 1.08; color: #0f172a; letter-spacing: -1px;">
                岗位检索
            </div>
            <div style="font-size: 13px; color: #64748b; margin-top: 8px; line-height: 1.65;">
                输入关键词后，可结合城市、方向、学历和薪资条件进行组合查询。
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    keyword = st.sidebar.text_input(
        "岗位关键词",
        "",
        key="filter_keyword",
        placeholder="如：医生、Java、销售、数据分析"
    )
    st.sidebar.caption("支持岗位名、公司、城市、技能、岗位方向的多关键词加权检索")

    st.sidebar.markdown(
        """
        <div style="height: 1px; background: rgba(148,163,184,0.32); margin: 18px 0 16px 0;"></div>
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(37,99,235,0.10);
            color: #1d4ed8;
            font-size: 14px;
            font-weight: 900;
            margin-bottom: 12px;
        ">高级检索</div>
        """,
        unsafe_allow_html=True
    )

    selected_cities = st.sidebar.multiselect(
        "选择城市",
        city_options,
        default=[],
        key="filter_cities"
    )

    selected_directions = st.sidebar.multiselect(
        "选择岗位方向",
        direction_options,
        default=[],
        key="filter_directions"
    )

    selected_degrees = st.sidebar.multiselect(
        "选择学历要求",
        degree_options,
        default=[],
        key="filter_degrees"
    )

    selected_salary = st.sidebar.slider(
        "选择平均薪资范围（元/月）",
        min_value=salary_min,
        max_value=salary_max,
        value=(salary_min, salary_max),
        step=500,
        key="filter_salary"
    )

    st.sidebar.markdown(
        """
        <div style="height: 1px; background: rgba(148,163,184,0.32); margin: 18px 0 14px 0;"></div>
        """,
        unsafe_allow_html=True
    )

    if st.sidebar.button("🔎 一键查询", use_container_width=True, key="query_by_filters_btn"):
        st.rerun()

    st.sidebar.button(
        "📊 最新就业总览",
        use_container_width=True,
        key="latest_market_btn",
        on_click=reset_to_latest_market
    )
    st.sidebar.caption("恢复全省、全方向、全学历视角，查看最新整体就业市场。")

    if st.sidebar.button("🔄 刷新本地数据", use_container_width=True, key="refresh_local_data_btn"):
        st.cache_data.clear()
        st.rerun()

    status_for_sidebar = get_stream_status(df)
    st.sidebar.caption(f"数据状态：准实时模拟｜最新批次：{status_for_sidebar['latest_time']}")

    st.sidebar.markdown(
        """
        <div style="height: 1px; background: rgba(148,163,184,0.32); margin: 18px 0 14px 0;"></div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.button(
        "← 返回封面页",
        use_container_width=True,
        key="back_to_cover_btn",
        on_click=back_to_cover
    )

    # 左侧筛选默认不选中：不选表示“全部”，避免一打开页面就被默认条件限制。
    city_mask = df["city"].isin(selected_cities) if selected_cities else pd.Series(True, index=df.index)
    direction_mask = df["job_direction"].isin(selected_directions) if selected_directions else pd.Series(True, index=df.index)
    degree_mask = df["degree_clean"].isin(selected_degrees) if selected_degrees else pd.Series(True, index=df.index)
    salary_mask = (df["salary_avg"] >= selected_salary[0]) & (df["salary_avg"] <= selected_salary[1])

    base_filtered_df = df[
        city_mask & direction_mask & degree_mask & salary_mask
    ].copy()

    # 关键词 / 筛选兜底逻辑：
    # 1）先按左侧条件严格筛选；
    # 2）如果严格筛选为空，不直接让右侧空白，而是自动放宽到“仅保留薪资范围”的全局数据；
    # 3）如果输入关键词，先在严格筛选内搜索；搜不到再在全局薪资范围内搜索；仍搜不到则展示薪资范围内数据，并给出提示。
    search_relaxed = False
    filter_relaxed = False
    no_keyword_match = False

    relaxed_pool = df[
        (df["salary_avg"] >= selected_salary[0]) &
        (df["salary_avg"] <= selected_salary[1])
    ].copy()

    if keyword.strip():
        filtered_df = weighted_keyword_search(base_filtered_df, keyword)

        if filtered_df.empty:
            filtered_df = weighted_keyword_search(relaxed_pool, keyword)
            search_relaxed = not filtered_df.empty

        if filtered_df.empty:
            # 兜底：关键词也搜不到时，不让页面变空，展示当前薪资范围下的整体市场作为参考
            filtered_df = relaxed_pool.copy()
            no_keyword_match = True
    else:
        filtered_df = base_filtered_df

        if filtered_df.empty:
            # 兜底：筛选组合太窄时，自动展示当前薪资范围下的全局市场
            filtered_df = relaxed_pool.copy()
            filter_relaxed = True

    if filtered_df.empty:
        # 极端兜底：如果薪资范围也没有数据，就回到全量数据，保证页面永远有内容展示
        filtered_df = df.copy()
        filter_relaxed = True

    if search_relaxed:
        st.info("当前关键词在左侧筛选条件内没有结果，系统已自动放宽城市/岗位方向/学历限制，仅保留薪资范围进行全库检索。")

    if filter_relaxed:
        st.warning("当前筛选组合没有匹配岗位，系统已自动放宽城市、岗位方向和学历限制，展示当前薪资范围下的整体就业市场，避免右侧页面空白。")

    if no_keyword_match:
        st.warning("当前关键词没有匹配岗位，系统已展示当前薪资范围下的整体市场作为参考。可以换成近义词，例如：医生可试试 医师、临床、医学、护理、药师。")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 数据总览",
        "💰 薪资学历",
        "🧠 技能画像",
        "📍 城市岗位",
        "🤖 岗位聚类",
        "🎯 求职匹配",
        "📁 数据中心"
    ])

    # =====================================================
    # Tab1 数据总览
    # =====================================================
    with tab1:
        total_jobs = len(filtered_df)
        total_companies = filtered_df["company_name"].nunique()
        total_cities = filtered_df["city"].nunique()
        avg_salary = filtered_df["salary_avg"].mean()
        high_salary_ratio = (filtered_df["salary_avg"] >= 12000).mean() * 100
        bachelor_ratio = filtered_df["degree_clean"].astype(str).str.contains("本科|硕士|博士").mean() * 100

        city_count = filtered_df["city"].value_counts().reset_index()
        city_count.columns = ["城市", "岗位数量"]

        direction_count = filtered_df["job_direction"].value_counts().reset_index()
        direction_count.columns = ["岗位方向", "岗位数量"]

        salary_by_city = (
            filtered_df.groupby("city")["salary_avg"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        salary_by_city.columns = ["城市", "平均薪资"]

        high_salary_city = (
            filtered_df[filtered_df["salary_avg"] >= 12000]["city"]
            .value_counts()
            .reset_index()
        )
        high_salary_city.columns = ["城市", "高薪岗位数"]

        top_city = city_count.iloc[0]["城市"]
        top_city_count = city_count.iloc[0]["岗位数量"]
        top_direction = direction_count.iloc[0]["岗位方向"]
        top_salary_city = salary_by_city.iloc[0]["城市"]
        top_salary_value = salary_by_city.iloc[0]["平均薪资"]

        skill_freq_overview = get_skill_frequency(filtered_df)
        if not skill_freq_overview.empty:
            top_skill = skill_freq_overview.iloc[0]["技能关键词"]
        else:
            top_skill = "暂无明显技能关键词"

        # =====================
        # 决策摘要区
        # =====================
        st.markdown("### 就业市场决策总览")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("岗位总量", f"{total_jobs}", "当前筛选范围")
        with k2:
            metric_card("覆盖企业", f"{total_companies}", "去重企业数量")
        with k3:
            metric_card("平均薪资", f"{avg_salary:,.0f} 元/月", "岗位平均薪资")
        with k4:
            metric_card("高薪岗位占比", f"{high_salary_ratio:.1f}%", "平均薪资 ≥ 12000 元")

        k5, k6, k7, k8 = st.columns(4)
        with k5:
            metric_card("招聘最热城市", top_city, f"{top_city_count} 条岗位")
        with k6:
            metric_card("主需求方向", top_direction, "岗位数量最多")
        with k7:
            metric_card("薪资优势城市", top_salary_city, f"{top_salary_value:,.0f} 元/月")
        with k8:
            metric_card("高频技能", top_skill, "简历优化参考")

        section_intro(
            "本页不只是展示岗位数量，而是从“哪里机会多、哪个方向需求强、哪里的薪资更高、学生应该补什么技能”四个角度，"
            "为浙江省大学生求职提供直接决策参考。"
        )

        # =====================
        # 实用建议区
        # =====================
        st.markdown("### 智能求职建议")

        advice_col1, advice_col2, advice_col3 = st.columns(3)

        with advice_col1:
            st.markdown(
                f"""
                <div class="insight-box">
                <b>城市选择建议：</b><br>
                当前岗位机会最集中的是 <b>{top_city}</b>，共采集到 <b>{top_city_count}</b> 条岗位。
                如果优先考虑岗位数量和选择空间，建议重点关注 {top_city} 及周边城市。
                </div>
                """,
                unsafe_allow_html=True
            )

        with advice_col2:
            st.markdown(
                f"""
                <div class="insight-box">
                <b>方向选择建议：</b><br>
                当前需求最明显的岗位方向是 <b>{top_direction}</b>。
                学生可以结合自身专业背景，优先准备该方向相关项目经历、技能证书和简历关键词。
                </div>
                """,
                unsafe_allow_html=True
            )

        with advice_col3:
            st.markdown(
                f"""
                <div class="insight-box">
                <b>能力提升建议：</b><br>
                当前样本中高频技能为 <b>{top_skill}</b>。
                建议将该类能力补充到简历、项目描述和面试表达中，提高岗位匹配度。
                </div>
                """,
                unsafe_allow_html=True
            )

        # =====================
        # 图表区：两行布局，更像 dashboard
        # =====================
        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

        row1_left, row1_right = st.columns([1.15, 1])

        with row1_left:
            chart_title("城市岗位机会排行榜")
            fig_city = px.bar(
                city_count,
                x="城市",
                y="岗位数量",
                text="岗位数量",
                color="岗位数量",
                color_continuous_scale=COLOR_SCALE_BLUE,
                template="simple_white"
            )
            fig_city.update_traces(
                textposition="outside",
                marker_line_width=0,
                hovertemplate="城市：%{x}<br>岗位数量：%{y}<extra></extra>"
            )
            fig_city.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_city, 390, showlegend=False), use_container_width=True)

        with row1_right:
            chart_title("岗位方向需求结构")
            direction_bar = direction_count.sort_values("岗位数量", ascending=True)
            fig_direction = px.bar(
                direction_bar,
                x="岗位数量",
                y="岗位方向",
                orientation="h",
                text="岗位数量",
                color="岗位数量",
                color_continuous_scale=COLOR_SCALE_BLUE,
                template="simple_white"
            )
            fig_direction.update_traces(
                textposition="outside",
                marker_line_width=0,
                hovertemplate="岗位方向：%{y}<br>岗位数量：%{x}<extra></extra>"
            )
            fig_direction.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_direction, 390, showlegend=False), use_container_width=True)

        row2_left, row2_right = st.columns([1, 1])

        with row2_left:
            chart_title("城市平均薪资对比")
            fig_salary_city = px.bar(
                salary_by_city,
                x="城市",
                y="平均薪资",
                text=salary_by_city["平均薪资"].round(0),
                color="平均薪资",
                color_continuous_scale=COLOR_SCALE_TEAL,
                template="simple_white"
            )
            fig_salary_city.update_traces(textposition="outside", marker_line_width=0)
            fig_salary_city.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_salary_city, 380, showlegend=False), use_container_width=True)

        with row2_right:
            chart_title("城市高薪岗位分布")
            if high_salary_city.empty:
                st.info("当前筛选条件下暂无高薪岗位。")
            else:
                fig_high_city = px.bar(
                    high_salary_city,
                    x="城市",
                    y="高薪岗位数",
                    text="高薪岗位数",
                    color="高薪岗位数",
                    color_continuous_scale=COLOR_SCALE_ORANGE,
                    template="simple_white"
                )
                fig_high_city.update_traces(textposition="outside", marker_line_width=0)
                fig_high_city.update_layout(coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig_high_city, 380, showlegend=False), use_container_width=True)

        insight_box(
            f"综合岗位数量、薪资水平和岗位方向来看，当前筛选范围内 <b>{top_city}</b> 的岗位机会最多，"
            f"<b>{top_salary_city}</b> 的平均薪资表现较高，主要需求方向为 <b>{top_direction}</b>。"
            f"如果学生追求机会数量，可优先关注 {top_city}；如果更关注薪资水平，可进一步查看 {top_salary_city} 的具体岗位。"
        )
    # =====================================================
    # Tab2 薪资学历
    # =====================================================
    with tab2:
        section_intro(
            "本页从薪资区间、学历要求和高薪岗位画像三个角度分析就业市场门槛与岗位价值。"
        )

        salary_count = filtered_df["salary_level"].value_counts().reset_index()
        salary_count.columns = ["薪资区间", "岗位数量"]

        salary_order = ["5K以下", "5K-8K", "8K-12K", "12K-20K", "20K以上", "薪资未知"]
        salary_count["薪资区间"] = pd.Categorical(
            salary_count["薪资区间"],
            categories=salary_order,
            ordered=True
        )
        salary_count = salary_count.sort_values("薪资区间")

        degree_count = filtered_df["degree_clean"].value_counts().reset_index()
        degree_count.columns = ["学历要求", "岗位数量"]

        left, right = st.columns(2)

        with left:
            chart_title("岗位薪资区间分布")
            fig_salary = px.bar(
                salary_count,
                x="薪资区间",
                y="岗位数量",
                text="岗位数量",
                color="岗位数量",
                color_continuous_scale=COLOR_SCALE_TEAL,
                template="simple_white"
            )
            fig_salary.update_traces(textposition="outside", marker_line_width=0)
            fig_salary.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_salary, 500, showlegend=False), use_container_width=True)

        with right:
            chart_title("不同学历要求岗位数量分布")
            fig_degree = px.bar(
                degree_count,
                x="学历要求",
                y="岗位数量",
                text="岗位数量",
                color="岗位数量",
                color_continuous_scale=COLOR_SCALE_PURPLE,
                template="simple_white"
            )
            fig_degree.update_traces(textposition="outside", marker_line_width=0)
            fig_degree.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_degree, 500, showlegend=False), use_container_width=True)

        chart_title("不同学历要求下的薪资分布")
        fig_degree_salary = px.box(
            filtered_df,
            x="degree_clean",
            y="salary_avg",
            color="degree_clean",
            points="outliers",
            template="simple_white",
            color_discrete_sequence=["#2563eb", "#38bdf8", "#8b5cf6", "#f97316", "#14b8a6"]
        )
        fig_degree_salary.update_layout(xaxis_title="学历要求", yaxis_title="平均薪资")
        st.plotly_chart(style_fig(fig_degree_salary, 560), use_container_width=True)

        high_df = filtered_df[filtered_df["salary_avg"] >= 12000].copy()

        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
        st.subheader("高薪岗位画像分析")

        if high_df.empty:
            st.info("当前筛选条件下暂无平均薪资 12000 元以上的高薪岗位。")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("高薪岗位数量", f"{len(high_df)}", "平均薪资 ≥ 12000 元")
            with c2:
                metric_card("高薪岗位平均薪资", f"{high_df['salary_avg'].mean():,.0f} 元/月", "高薪样本均值")
            with c3:
                metric_card("最高岗位薪资", f"{high_df['salary_avg'].max():,.0f} 元/月", "当前筛选范围")

            high_direction = high_df["job_direction"].value_counts().reset_index()
            high_direction.columns = ["岗位方向", "高薪岗位数量"]

            chart_title("高薪岗位方向分布")
            fig_high_direction = px.bar(
                high_direction.sort_values("高薪岗位数量", ascending=True),
                x="高薪岗位数量",
                y="岗位方向",
                orientation="h",
                text="高薪岗位数量",
                color="高薪岗位数量",
                color_continuous_scale=COLOR_SCALE_ORANGE,
                template="simple_white"
            )
            fig_high_direction.update_traces(textposition="outside", marker_line_width=0)
            fig_high_direction.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_high_direction, 520, showlegend=False), use_container_width=True)

            insight_box(
                f"当前筛选条件下，高薪岗位共有 <b>{len(high_df)}</b> 条，"
                f"高薪岗位平均薪资约为 <b>{high_df['salary_avg'].mean():,.0f}</b> 元/月。"
                f"高薪岗位通常与技术能力、学历门槛、行业方向和城市发展水平有关。"
            )

    # =====================================================
    # Tab3 技能画像
    # =====================================================
    with tab3:
        section_intro(
            "本页通过技能关键词频率和词云展示岗位对具体能力的需求，可用于指导学生优化简历和提升技能。"
        )

        skill_freq = get_skill_frequency(filtered_df)

        if skill_freq.empty:
            st.info("当前筛选条件下没有提取到技能关键词。")
        else:
            top_skills = skill_freq.head(20)

            chart_title("岗位技能关键词 Top20")
            fig_skill = px.bar(
                top_skills.sort_values("出现次数", ascending=True),
                x="出现次数",
                y="技能关键词",
                orientation="h",
                text="出现次数",
                color="出现次数",
                color_continuous_scale=COLOR_SCALE_TEAL,
                template="simple_white"
            )
            fig_skill.update_traces(textposition="outside", marker_line_width=0)
            fig_skill.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_skill, 620, showlegend=False), use_container_width=True)

            top_skill_name = top_skills.iloc[0]["技能关键词"]
            top_skill_count = top_skills.iloc[0]["出现次数"]

            insight_box(
                f"当前筛选条件下，出现频率最高的技能关键词是 <b>{top_skill_name}</b>，"
                f"共出现 <b>{top_skill_count}</b> 次。技能关键词能够反映就业市场对具体能力的需求方向。"
            )

            left_cloud, right_cloud = st.columns([1.05, 0.95])

            with left_cloud:
                chart_title("技能关键词矩阵图")
                matrix_fig = make_skill_matrix(skill_freq)
                if matrix_fig is not None:
                    st.plotly_chart(matrix_fig, use_container_width=True)

            with right_cloud:
                chart_title("岗位能力类别结构")
                category_fig = make_skill_category_chart(skill_freq)
                if category_fig is not None:
                    st.plotly_chart(category_fig, use_container_width=True)

            insight_box(
                "技能分析不再使用传统词云，改用关键词矩阵和能力类别结构图，避免中文字体缺失导致方框问题，"
                "同时能更清楚地展示市场到底需要哪几类能力。"
            )

    # =====================================================
    # Tab4 城市岗位
    # =====================================================
    with tab4:
        section_intro(
            "本页通过城市—岗位方向交叉热力图，展示浙江不同城市在岗位方向上的结构差异。"
        )

        heat_df = pd.crosstab(filtered_df["city"], filtered_df["job_direction"])

        if heat_df.empty:
            st.info("当前筛选条件下无法生成城市—岗位方向热力图。")
        else:
            chart_title("城市—岗位方向需求热力图")
            fig_heat = px.imshow(
                heat_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=COLOR_SCALE_BLUE,
                template="simple_white"
            )
            fig_heat.update_layout(
                height=660,
                plot_bgcolor="rgba(255,255,255,0)",
                paper_bgcolor="rgba(255,255,255,0)",
                xaxis_title="岗位方向",
                yaxis_title="城市",
                font=dict(size=13, color="#334155"),
                margin=dict(l=35, r=35, t=60, b=50)
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            max_city, max_direction = heat_df.stack().idxmax()
            max_value = heat_df.stack().max()

            insight_box(
                f"岗位需求最集中的组合是 <b>{max_city} - {max_direction}</b>，"
                f"共有 <b>{max_value}</b> 条岗位。该结果可以帮助学生识别不同城市的优势就业方向。"
            )

            st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
            chart_title("浙江就业空间分布图")
            fig_city_map = draw_city_map(filtered_df)
            city_geo_df = build_city_geo_df(filtered_df)

            if fig_city_map is None or city_geo_df.empty:
                st.info("当前筛选条件下没有可用于地图展示的城市坐标数据。")
            else:
                st.plotly_chart(fig_city_map, use_container_width=True)
                top_geo_city = city_geo_df.iloc[0]["城市"]
                top_geo_count = int(city_geo_df.iloc[0]["岗位数量"])
                top_salary_row = city_geo_df.sort_values("平均薪资", ascending=False).iloc[0]
                insight_box(
                    f"从空间分布看，当前岗位数量最多的城市是 <b>{top_geo_city}</b>，"
                    f"共 <b>{top_geo_count}</b> 条岗位；平均薪资表现较高的城市是 "
                    f"<b>{top_salary_row['城市']}</b>，约 <b>{top_salary_row['平均薪资']:,.0f}</b> 元/月。"
                    f"地图气泡大小表示岗位数量，颜色深浅表示平均薪资，可直观看出浙江就业机会的城市集聚与扩散格局。"
                )

    # =====================================================
    # Tab5 岗位聚类
    # =====================================================
    with tab5:
        section_intro(
            "本页基于薪资、学历、技能数量、岗位方向、城市和企业性质等多维特征，对岗位进行 K-Means 聚类，形成可解释的岗位画像。"
        )

        cluster_k = st.slider(
            "选择聚类数量 K",
            min_value=4,
            max_value=8,
            value=6,
            step=1
        )

        cluster_df, cluster_model = run_kmeans(filtered_df, n_clusters=cluster_k)

        if cluster_df.empty:
            st.info("当前筛选条件下无法进行聚类分析。")
        else:
            cluster_summary = cluster_df.groupby("cluster").agg(
                岗位数量=("job_name", "count"),
                平均薪资=("salary_avg", "mean"),
                薪资中位数=("salary_avg", "median"),
                高薪岗位占比=("is_high_salary", "mean"),
                平均技能数=("skill_count", "mean"),
                主要城市=("city", lambda x: x.value_counts().idxmax()),
                主要岗位方向=("job_direction", lambda x: x.value_counts().idxmax()),
                主要学历要求=("degree_clean", lambda x: x.value_counts().idxmax())
            ).reset_index()

            cluster_summary["高薪岗位占比"] = cluster_summary["高薪岗位占比"] * 100
            cluster_summary["岗位画像名称"] = make_cluster_display_names(cluster_summary)

            name_map = dict(zip(cluster_summary["cluster"], cluster_summary["岗位画像名称"]))
            cluster_df["岗位画像名称"] = cluster_df["cluster"].map(name_map)

            st.dataframe(
                cluster_summary[
                    [
                        "cluster", "岗位画像名称", "岗位数量", "平均薪资", "薪资中位数",
                        "高薪岗位占比", "平均技能数", "主要城市", "主要岗位方向", "主要学历要求"
                    ]
                ],
                use_container_width=True
            )

            chart_title("岗位聚类散点图：多维特征降维展示")
            fig_cluster_points = px.scatter(
                cluster_df,
                x="聚类横轴",
                y="聚类纵轴",
                color="岗位画像名称",
                size="salary_avg",
                hover_data={
                    "job_name": True,
                    "company_name": True,
                    "city": True,
                    "salary_avg": ":,.0f",
                    "degree_clean": True,
                    "job_direction": True,
                    "skill_keywords": True,
                    "聚类横轴": False,
                    "聚类纵轴": False,
                },
                size_max=26,
                opacity=0.72,
                color_discrete_sequence=["#2563eb", "#38bdf8", "#f97316", "#8b5cf6", "#14b8a6", "#ef4444", "#0f766e"],
                labels={
                    "岗位画像名称": "岗位画像",
                    "聚类横轴": "岗位综合特征轴 1",
                    "聚类纵轴": "岗位综合特征轴 2",
                    "salary_avg": "平均薪资"
                },
                template="simple_white"
            )
            fig_cluster_points.update_traces(marker=dict(line=dict(width=0.7, color="white")))
            st.plotly_chart(style_fig(fig_cluster_points, 560), use_container_width=True)

            left_cluster, right_cluster = st.columns([1.05, 0.95])

            with left_cluster:
                chart_title("岗位画像气泡图：数量 × 薪资 × 技能复杂度")
                fig_cluster = px.scatter(
                    cluster_summary,
                    x="平均薪资",
                    y="平均技能数",
                    size="岗位数量",
                    color="岗位画像名称",
                    text="岗位画像名称",
                    hover_data={
                        "岗位画像名称": True,
                        "岗位数量": True,
                        "平均薪资": ":,.0f",
                        "薪资中位数": ":,.0f",
                        "高薪岗位占比": ":.1f",
                        "平均技能数": ":.2f",
                        "主要城市": True,
                        "主要岗位方向": True,
                        "主要学历要求": True,
                    },
                    template="simple_white",
                    size_max=78,
                    color_discrete_sequence=["#2563eb", "#38bdf8", "#f97316", "#8b5cf6", "#14b8a6", "#ef4444", "#0f766e"],
                    labels={
                        "平均薪资": "平均薪资（元/月）",
                        "平均技能数": "平均技能数",
                        "岗位画像名称": "岗位画像"
                    }
                )
                fig_cluster.update_traces(
                    textposition="top center",
                    marker=dict(opacity=0.82, line=dict(width=1.2, color="white"))
                )
                st.plotly_chart(style_fig(fig_cluster, 500), use_container_width=True)

            with right_cluster:
                chart_title("岗位画像数量结构")
                fig_cluster_bar = px.bar(
                    cluster_summary.sort_values("岗位数量", ascending=True),
                    x="岗位数量",
                    y="岗位画像名称",
                    orientation="h",
                    text="岗位数量",
                    color="平均薪资",
                    color_continuous_scale=COLOR_SCALE_BLUE,
                    template="simple_white",
                    labels={"岗位画像名称": "岗位画像", "岗位数量": "岗位数量", "平均薪资": "平均薪资"}
                )
                fig_cluster_bar.update_traces(textposition="outside", marker_line_width=0)
                fig_cluster_bar.update_layout(coloraxis_colorbar=dict(title="平均薪资"))
                st.plotly_chart(style_fig(fig_cluster_bar, 500, showlegend=False), use_container_width=True)

            st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
            left_more, right_more = st.columns([1.08, 0.92])

            with left_more:
                chart_title("岗位画像 × 城市分布")
                top_cities_for_cluster = cluster_df["city"].value_counts().head(8).index.tolist()
                cluster_city_df = cluster_df[cluster_df["city"].isin(top_cities_for_cluster)].copy()
                cluster_city = (
                    cluster_city_df.groupby(["岗位画像名称", "city"])
                    .size()
                    .reset_index(name="岗位数量")
                )
                fig_cluster_city = px.bar(
                    cluster_city,
                    x="岗位数量",
                    y="岗位画像名称",
                    color="city",
                    orientation="h",
                    text="岗位数量",
                    template="simple_white",
                    labels={"city": "城市", "岗位画像名称": "岗位画像"},
                    color_discrete_sequence=["#2563eb", "#38bdf8", "#14b8a6", "#8b5cf6", "#f97316", "#ef4444", "#0f766e", "#64748b"]
                )
                fig_cluster_city.update_traces(textposition="inside", marker_line_width=0)
                st.plotly_chart(style_fig(fig_cluster_city, 520), use_container_width=True)

            with right_more:
                chart_title("岗位画像 × 学历要求热力图")
                cluster_degree = pd.crosstab(cluster_df["岗位画像名称"], cluster_df["degree_clean"])
                fig_cluster_degree = px.imshow(
                    cluster_degree,
                    text_auto=True,
                    color_continuous_scale=COLOR_SCALE_PURPLE,
                    aspect="auto",
                    labels=dict(x="学历要求", y="岗位画像", color="岗位数量")
                )
                fig_cluster_degree.update_layout(
                    height=520,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,0)",
                    font=dict(size=12, color="#334155"),
                )
                st.plotly_chart(fig_cluster_degree, use_container_width=True)

            insight_box(
                "岗位画像已按聚类序号、主要城市和主要方向生成唯一名称，因此图例会展示完整的多个岗位群体；"
                "城市分布和学历热力图可以进一步说明不同岗位画像在空间和门槛上的差异。"
            )

            largest_cluster = cluster_summary.sort_values("岗位数量", ascending=False).iloc[0]
            salary_cluster = cluster_summary.sort_values("平均薪资", ascending=False).iloc[0]

            insight_box(
                f"岗位数量最多的聚类画像是 <b>{largest_cluster['岗位画像名称']}</b>，"
                f"共有 <b>{largest_cluster['岗位数量']}</b> 条岗位，主要集中在 <b>{largest_cluster['主要城市']}</b>，"
                f"主要方向为 <b>{largest_cluster['主要岗位方向']}</b>；"
                f"薪资最高的岗位画像是 <b>{salary_cluster['岗位画像名称']}</b>，"
                f"平均薪资约为 <b>{salary_cluster['平均薪资']:,.0f}</b> 元/月。"
            )


    # =====================================================
    # Tab6 求职匹配
    # =====================================================
    with tab6:
        section_intro(
            "用户可以输入个人求职条件，系统将基于当前岗位数据进行匹配，并给出岗位列表、薪资参考和求职建议。"
        )

        match_col1, match_col2 = st.columns(2)

        with match_col1:
            target_city = st.selectbox(
                "意向城市",
                ["不限"] + sorted(df["city"].dropna().unique().tolist())
            )

            target_degree = st.selectbox(
                "最高学历",
                ["不限"] + sorted(df["degree_clean"].dropna().unique().tolist())
            )

            target_direction = st.selectbox(
                "意向岗位方向",
                ["不限"] + sorted(df["job_direction"].dropna().unique().tolist())
            )

        with match_col2:
            expected_salary = st.slider(
                "期望最低平均薪资（元/月）",
                min_value=int(df["salary_avg"].min()),
                max_value=int(df["salary_avg"].max()),
                value=6000,
                step=500
            )

            user_skills = st.text_input(
                "我的技能关键词",
                "Python, Excel, 数据分析"
            )

        matched_jobs = df.copy()

        if target_city != "不限":
            matched_jobs = matched_jobs[matched_jobs["city"] == target_city]

        if target_degree != "不限":
            matched_jobs = matched_jobs[matched_jobs["degree_clean"] == target_degree]

        if target_direction != "不限":
            matched_jobs = matched_jobs[matched_jobs["job_direction"] == target_direction]

        matched_jobs = matched_jobs[matched_jobs["salary_avg"] >= expected_salary]

        if user_skills.strip():
            searched_jobs = weighted_keyword_search(matched_jobs, user_skills)
            if not searched_jobs.empty:
                matched_jobs = searched_jobs
            else:
                matched_jobs = matched_jobs.copy()
                matched_jobs["search_score"] = 0

        st.markdown(
            f"""
            <div class="insight-box">
            <b>匹配建议：</b>{generate_job_advice(user_skills, matched_jobs)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("匹配岗位 Top 30")

        if matched_jobs.empty:
            st.info("暂无匹配岗位，请尝试降低期望薪资或放宽筛选条件。")
        else:
            st.dataframe(
                matched_jobs[
                    [
                        "job_name", "company_name", "city", "salary_avg",
                        "degree_clean", "job_direction", "skill_keywords",
                        "company_property", "source"
                    ]
                ].sort_values(by="salary_avg", ascending=False).head(30),
                use_container_width=True
            )

    # =====================================================
    # Tab7 数据中心
    # =====================================================
    with tab7:
        section_intro(
            "本页展示清洗后的岗位数据明细，可用于验证数据来源和筛选分析结果。"
        )

        show_cols = [
            "job_name", "company_name", "city", "salary_min", "salary_max",
            "salary_avg", "degree_clean", "job_direction", "skill_keywords",
            "company_property", "source", "data_type", "stream_batch", "stream_time"
        ]
        show_cols = [col for col in show_cols if col in filtered_df.columns]

        st.dataframe(
            filtered_df[show_cols].sort_values(by="salary_avg", ascending=False),
            use_container_width=True,
            height=650
        )

        csv_data = filtered_df[show_cols].to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="下载当前筛选数据 CSV",
            data=csv_data,
            file_name="zhejiang_jobs_filtered.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()