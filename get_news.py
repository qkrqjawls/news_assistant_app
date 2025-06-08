import os
import requests
from datetime import datetime, timedelta, timezone
import json
from email.utils import parsedate_to_datetime

NEWSDATAIO_API_KEY = os.environ.get("NEWSDATAIO_API_KEY", "your-api-key")  # ← 반드시 발급받은 실제 키로 대체하세요

def parse_pubdate(dt_str):
    """
    pubDate가 RFC2822 형식이면 그대로 파싱,
    ISO8601 형식(YYYY-MM-DD HH:MM:SS)이면 UTC 기준으로 처리
    """
    try:
        return parsedate_to_datetime(dt_str).astimezone(timezone.utc)
    except Exception:
        return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)

def fetch_recent_kr_news(minutes=30, now_utc=datetime.now(timezone.utc), per_page=50):
    URL = "https://newsdata.io/api/1/latest"

    if not isinstance(now_utc, datetime): now_utc = datetime.now(timezone.utc)

    cutoff = now_utc - timedelta(minutes=minutes)

    seen_links = set()
    results = []
    page_token = None
    page_count = 0

    while True:
        params = {
            "apikey": NEWSDATAIO_API_KEY,
            "country": "kr",
            "size": per_page
        }
        if page_token:
            params["page"] = page_token

        resp = requests.get(URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("results", [])
        if not articles:
            break

        page_count += 1
        new_found = False

        for art in articles:
            dt_raw = art.get("pubDate")
            if not dt_raw:
                continue

            try:
                pub = parse_pubdate(dt_raw)
            except Exception:
                continue

            if pub < cutoff:
                continue

            link = art.get("link") or art.get("guid")
            if link and link not in seen_links:
                seen_links.add(link)
                results.append(art)
                new_found = True

        if not new_found:
            print(f"[INFO] No articles within the last {minutes} minutes on page {page_count}. Stopping early.")
            break

        page_token = data.get("nextPage")
        if not page_token:
            break

    print(f"[DONE] Fetched {len(results)} articles from {page_count} page(s).")
    return results

def serialize(value):
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return value

def parse_datetime(dtstr):
    try:
        return datetime.fromisoformat(dtstr.replace("Z", "+00:00"))
    except:
        return None
