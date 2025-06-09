# utils.py

import os
import json
import re
import requests


# .env 혹은 셸 환경변수에서 CHAT_GPT_API_KEY를 읽어옵니다.

CHAT_GPT_API_KEY = os.environ.get("CHAT_GPT_API_KEY", "appuser")

def load_all_json_from_folder(folder_path: str) -> list[dict]:
    """
    newsdata/ 폴더 안의 모든 .json 파일을 로드해 dict 리스트로 반환
    """
    items = []
    for fn in os.listdir(folder_path):
        if fn.endswith(".json"):
            with open(os.path.join(folder_path, fn), "r", encoding="utf-8") as f:
                items.append(json.load(f))
    return items

def preprocess_text(title: str, body: str) -> str:
    """
    제목+본문 합친 뒤 간단 정제(특수문자 제거, 공백 정리)
    """
    text = (title or "") + " " + (body or "")
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def summarize_with_openai(cluster_texts: list[str]) -> dict:
    combined = "\n".join(cluster_texts)
    if len(combined) > 2000:
        combined = combined[:2000] + "…"

    def ask_gpt(prompt):
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {CHAT_GPT_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.7
            }
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    title = ask_gpt(
        "다음 뉴스 기사들의 공통 주제를 15~20자 이내의 간결한 '이슈 제목'으로 제시하세요.\n"
        f"----------------\n{combined}\n----------------\n이슈 제목:"
    )
    summary = ask_gpt(
        "다음 뉴스 기사들을 2~3문단으로 요약해 주세요.\n"
        f"----------------\n{combined}\n----------------"
    )

    return {
        "issue_name": title,
        "issue_summary": summary
    }
