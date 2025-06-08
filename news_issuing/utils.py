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
    """
    한 클러스터(이슈)에 속한 기사 텍스트 리스트에서
    '이슈 이름'과 '이슈 요약'을 뽑아 반환.
    """
    if not CHAT_GPT_API_KEY:
        raise RuntimeError("환경변수에 CHAT_GPT_API_KEY를 설정하세요.")
    # 텍스트 합치기 & 길이 제한
    combined = "\n".join(cluster_texts)
    if len(combined) > 2000:
        combined = combined[:2000] + "…"

    prompt = (
        "아래 여러 뉴스 기사로부터\n"
        "1) 이슈 이름(15~20자)\n"
        "2) 이슈 요약(2~3문단)\n"
        "형태로 답해주세요.\n"
        "----------------\n"
        f"{combined}\n"
        "----------------\n"
    )

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {CHAT_GPT_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "당신은 뉴스 요약 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.7
        }
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # 간단 파싱: “이슈 이름:…” “이슈 요약:…”
    issue_name, issue_summary = "", ""
    for line in content.splitlines():
        if line.startswith("이슈 이름"):
            issue_name = line.split(":", 1)[1].strip()
        elif line.startswith("이슈 요약"):
            issue_summary = line.split(":", 1)[1].strip()
    # 파싱 실패 시 전체를 요약으로
    return {
        "issue_name": issue_name,
        "issue_summary": issue_summary or content
    }
