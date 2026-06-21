import re
from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw_jobs.csv")
CLEAN_PATH = Path("data/clean_jobs.csv")


SKILL_WORDS = [
    "Python", "Java", "SQL", "Excel", "PPT", "Word",
    "数据分析", "数据挖掘", "机器学习", "深度学习",
    "Tableau", "PowerBI", "Linux", "MySQL", "Oracle",
    "HTML", "CSS", "JavaScript", "Vue", "React",
    "Spring", "SpringBoot", "电商", "运营", "新媒体",
    "销售", "外贸", "财务", "会计", "行政", "机械",
    "CAD", "PLC", "英语", "沟通", "文案", "短视频",
    "直播", "客服", "教师", "设计", "PS", "剪辑"
]


def normalize_degree(degree):
    degree = str(degree)

    if "博士" in degree:
        return "博士及以上"
    if "硕士" in degree:
        return "硕士及以上"
    if "本科" in degree:
        return "本科及以上"
    if "专科" in degree or "大专" in degree:
        return "专科及以上"
    if "不限" in degree:
        return "不限"

    return "其他"


def get_degree_level(degree):
    mapping = {
        "不限": 0,
        "专科及以上": 1,
        "本科及以上": 2,
        "硕士及以上": 3,
        "博士及以上": 4,
        "其他": 0
    }
    return mapping.get(degree, 0)


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


def extract_skills(text):
    text = str(text)
    found = []

    for skill in SKILL_WORDS:
        if skill.lower() in text.lower():
            found.append(skill)

    return "、".join(sorted(set(found)))


def classify_direction(text):
    text = str(text)

    if any(word in text for word in ["Java", "Python", "软件", "开发", "前端", "后端", "算法", "数据分析", "SQL"]):
        return "技术/数据类"
    if any(word in text for word in ["运营", "新媒体", "电商", "短视频", "直播", "文案"]):
        return "运营/传媒类"
    if any(word in text for word in ["销售", "客户", "客服", "市场", "商务"]):
        return "销售/市场类"
    if any(word in text for word in ["财务", "会计", "审计", "出纳"]):
        return "财务/审计类"
    if any(word in text for word in ["行政", "人事", "人力资源", "助理"]):
        return "行政/人力类"
    if any(word in text for word in ["机械", "电气", "制造", "工程师", "CAD", "PLC"]):
        return "制造/工程类"
    if any(word in text for word in ["教师", "英语", "教育", "培训"]):
        return "教育/培训类"
    if any(word in text for word in ["外贸", "跨境", "进出口"]):
        return "外贸/跨境类"

    return "其他类"


def main():
    if not RAW_PATH.exists():
        print("没有找到 data/raw_jobs.csv，请先运行爬虫。")
        return

    df = pd.read_csv(RAW_PATH)

    print("原始数据量：", len(df))

    # 去重
    if "job_id" in df.columns:
        df = df.drop_duplicates(subset=["job_id"])
    else:
        df = df.drop_duplicates()

    # 必要字段填充
    for col in ["job_name", "company_name", "city", "degree", "major", "skills", "keyword"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    # 薪资字段转数值
    for col in ["salary_min", "salary_max", "salary_avg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 没有 salary_avg 的，用 min/max 重新算
    df["salary_avg"] = df["salary_avg"].fillna(
        (df["salary_min"] + df["salary_max"]) / 2
    )

    # 删除完全没有薪资的岗位
    df = df.dropna(subset=["salary_avg"])

    # 学历标准化
    df["degree_clean"] = df["degree"].apply(normalize_degree)
    df["degree_level"] = df["degree_clean"].apply(get_degree_level)

    # 薪资等级
    df["salary_level"] = df["salary_avg"].apply(get_salary_level)

    # 文本合并
    df["combined_text"] = (
        df["job_name"].astype(str) + " " +
        df["major"].astype(str) + " " +
        df["skills"].astype(str) + " " +
        df["keyword"].astype(str)
    )

    # 技能提取
    df["skill_keywords"] = df["combined_text"].apply(extract_skills)

    # 岗位方向分类
    df["job_direction"] = df["combined_text"].apply(classify_direction)

    # 高薪岗位标记
    df["is_high_salary"] = df["salary_avg"].apply(lambda x: 1 if x >= 12000 else 0)

    # 日期字段处理
    for col in ["publish_date", "update_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 保存
    df.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")

    print("清洗后数据量：", len(df))
    print("已保存到：", CLEAN_PATH)
    print(df.head())


if __name__ == "__main__":
    main()