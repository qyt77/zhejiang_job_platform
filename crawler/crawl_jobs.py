import time
import random
from pathlib import Path

import requests
import pandas as pd


# 这里用你 test_ajax.py 里已经成功的接口地址
AJAX_URL = "https://job.ncss.cn/student/jobs/jobslist/ajax/"

COOKIE_PATH = Path("crawler/cookie.txt")
OUTPUT_PATH = Path("data/raw_jobs.csv")


ZHEJIANG_CITIES = {
    "杭州": "330100",
    "宁波": "330200",
    "温州": "330300",
    "嘉兴": "330400",
    "湖州": "330500",
    "绍兴": "330600",
    "金华": "330700",
    "衢州": "330800",
    "舟山": "330900",
    "台州": "331000",
    "丽水": "331100",
}


JOB_KEYWORDS = [
    "工程师",
    "教师",
    "销售",
    "运营",
    "助理",
    "会计",
    "行政",
    "技术",
    "外贸",
    "机械",
    "电商",
    "客服",
    "财务",
    "实习",
    "管培生",
    "Java",
    "Python",
    "新媒体",
    "设计",
    "采购",
    "人事"
]


def load_cookie():
    if not COOKIE_PATH.exists():
        raise FileNotFoundError("没有找到 crawler/cookie.txt，请先保存浏览器 Cookie。")
    return COOKIE_PATH.read_text(encoding="utf-8").strip()


def build_headers(cookie_text):
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/149.0.0.0 Safari/537.36 Edg/149.0.0.0"
        ),
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://job.ncss.cn/student/jobs/index.html",
        "Accept": "*/*",
        "Cookie": cookie_text,
    }


def parse_timestamp(ts):
    """
    平台返回的是毫秒时间戳，转换成日期。
    """
    try:
        if ts is None:
            return ""
        return pd.to_datetime(int(ts), unit="ms").strftime("%Y-%m-%d")
    except Exception:
        return ""


def parse_salary(value):
    """
    平台 lowMonthPay / highMonthPay 通常以 K 为单位，如 11.0 表示 11K。
    转成元：11.0 -> 11000
    """
    try:
        if value is None or value == "":
            return None
        return float(value) * 1000
    except Exception:
        return None


def fetch_jobs(area_code, city_name, keyword, page, limit=10):
    """
    请求岗位接口。
    page 从 0 开始，offset = page * limit。
    """
    cookie_text = load_cookie()
    headers = build_headers(cookie_text)

    params = {
        "jobType": "",
        "areaCode": area_code,
        "jobName": keyword,
        "monthPay": "",
        "industrySectors": "",
        "property": "",
        "categoryCode": "",
        "memberLevel": "",
        "recruitType": "0",
        "offset": str(page * limit),
        "limit": str(limit),
        "keyUnits": "",
        "degreeCode": "",
        "sourcesName": "0",
        "sourcesType": "",
        "_": str(int(time.time() * 1000)),
    }

    response = requests.get(
        AJAX_URL,
        headers=headers,
        params=params,
        timeout=20,
    )
    response.encoding = "utf-8"

    if response.status_code != 200:
        print(f"请求失败：{city_name}-{keyword}-第{page + 1}页，状态码：{response.status_code}")
        return []

    try:
        data = response.json()
    except Exception:
        print(f"返回不是 JSON：{city_name}-{keyword}-第{page + 1}页")
        print(response.text[:200])
        return []

    if not data.get("flag"):
        print(f"接口返回失败：{city_name}-{keyword}-第{page + 1}页")
        print(data)
        return []

    job_list = data.get("data", {}).get("list", [])
    if not job_list:
        return []

    rows = []

    for item in job_list:
        low_salary = parse_salary(item.get("lowMonthPay"))
        high_salary = parse_salary(item.get("highMonthPay"))

        if low_salary is not None and high_salary is not None:
            salary_avg = (low_salary + high_salary) / 2
        elif low_salary is not None:
            salary_avg = low_salary
        elif high_salary is not None:
            salary_avg = high_salary
        else:
            salary_avg = None

        rows.append({
            "job_name": item.get("jobName", ""),
            "company_name": item.get("recName", ""),
            "city": city_name,
            "area_code_name": item.get("areaCodeName", ""),
            "industry": item.get("industrySectorsName", ""),
            "salary_min": low_salary,
            "salary_max": high_salary,
            "salary_avg": salary_avg,
            "degree": item.get("degreeName", ""),
            "head_count": item.get("headCount", ""),
            "company_size": item.get("recScale", ""),
            "company_property": item.get("recProperty", ""),
            "major": item.get("major", ""),
            "skills": item.get("recTags", ""),
            "publish_date": parse_timestamp(item.get("publishDate")),
            "update_date": parse_timestamp(item.get("updateDate")),
            "job_id": item.get("jobId", ""),
            "recruit_type": item.get("recruitType", ""),
            "source": item.get("sourcesNameCh", "") or "国家大学生就业服务平台",
            "keyword": keyword,
        })

    return rows


def crawl_all(max_pages_per_keyword=3, limit=10):
    """
    正式采集：
    浙江 11 个城市 × 多个关键词 × 每个关键词若干页。
    """
    all_rows = []

    for city_name, area_code in ZHEJIANG_CITIES.items():
        for keyword in JOB_KEYWORDS:
            for page in range(max_pages_per_keyword):
                print(f"正在采集：{city_name} | {keyword} | 第 {page + 1} 页")

                rows = fetch_jobs(
                    area_code=area_code,
                    city_name=city_name,
                    keyword=keyword,
                    page=page,
                    limit=limit,
                )

                if not rows:
                    print("本页无数据，停止该关键词翻页。")
                    break

                all_rows.extend(rows)

                # 礼貌等待，避免请求太频繁
                time.sleep(random.uniform(0.8, 1.6))

    return all_rows


def save_jobs(rows):
    Path("data").mkdir(exist_ok=True)

    df = pd.DataFrame(rows)

    if df.empty:
        print("没有采集到数据。")
        return

    # 去重
    if "job_id" in df.columns:
        df = df.drop_duplicates(subset=["job_id"])
    else:
        df = df.drop_duplicates()

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"采集完成，共 {len(df)} 条岗位数据。")
    print(f"已保存到：{OUTPUT_PATH}")
    print("=" * 60)

    print(df.head())


if __name__ == "__main__":
    rows = crawl_all(
        max_pages_per_keyword=4,  # 先小规模测试，每个关键词爬 2 页
        limit=10
    )
    save_jobs(rows)