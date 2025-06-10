from collections import defaultdict
from news_issuing.utils import summarize_with_openai

def group_by_cluster(items):
    groups = defaultdict(list)
    for it in items:
        groups[it.get("cluster_id", -1)].append(it)
    return groups

from typing import List, Dict

def summarize_and_save(clustered_data: List[List[dict]]) -> List[Dict[str, str]]:
    if not clustered_data:
        return []

    issue_list = [dict() for _ in range(len(clustered_data))]
    for cid, lst in enumerate(clustered_data):
        if not lst:
            issue_list[cid] = {"issue_name": "빈 클러스터", "issue_summary": "내용 없음"}
            continue
        snippets = [
            (str(it.get("title") or "") + " " + str(it.get("content") or ""))[:500 if idx != 0 else 2000]
            for idx, it in enumerate(lst[:10])
        ]
        try:
            issue_list[cid] = summarize_with_openai(snippets)
        except Exception as e:
            issue_list[cid] = {
                "issue_name": "요약 실패",
                "issue_summary": "OpenAI 요약 중 오류 발생"
            }
            print(f"[!] summarize_with_openai failed for cluster {cid}: {e}")

    return issue_list

if __name__ == "__main__":
    pass
