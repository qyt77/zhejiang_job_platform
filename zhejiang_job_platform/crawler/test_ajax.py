import requests
from pathlib import Path


url = "https://job.ncss.cn/student/jobs/jobslist/ajax/"

cookie_path = Path("crawler/cookie.txt")

if not cookie_path.exists():
    print("没有找到 crawler/cookie.txt，请先把浏览器里的 Cookie 保存进去。")
    raise SystemExit

cookie_text = cookie_path.read_text(encoding="utf-8").strip()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36 Edg/149.0.0.0",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://job.ncss.cn/student/jobs/index.html",
    "Accept": "*/*",
    "Cookie": cookie_text
}

params = {
    "jobType": "",
    "areaCode": "330304",      # 温州市瓯海区
    "jobName": "",
    "monthPay": "",
    "industrySectors": "",
    "property": "",
    "categoryCode": "",
    "memberLevel": "",
    "recruitType": "0",
    "offset": "0",
    "limit": "10",
    "keyUnits": "",
    "degreeCode": "",
    "sourcesName": "0",
    "sourcesType": ""
}

response = requests.get(url, headers=headers, params=params, timeout=15)
response.encoding = "utf-8"

print("状态码：", response.status_code)
print("返回长度：", len(response.text))
print("返回前1000字符：")
print(response.text[:1000])

Path("data").mkdir(exist_ok=True)

with open("data/ajax_test_result.html", "w", encoding="utf-8") as f:
    f.write(response.text)

print("已保存到 data/ajax_test_result.html")