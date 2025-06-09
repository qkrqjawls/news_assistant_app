from collections import defaultdict
from news_issuing.utils import summarize_with_openai

def group_by_cluster(items):
    groups = defaultdict(list)
    for it in items:
        groups[it.get("cluster_id", -1)].append(it)
    return groups

def summarize_and_save(clustered_data: list):
    if not clustered_data:
        return []

    issue_map = [dict() for _ in range(len(clustered_data))]
    for cid, lst in enumerate(clustered_data):
        if not lst:
            issue_map[cid] = {"issue_name": "빈 클러스터", "issue_summary": "내용 없음"}
            continue
        snippets = [
            (str(it.get("title") or "") + " " + str(it.get("content") or ""))[:200]
            for it in lst
        ]
        issue_map[cid] = summarize_with_openai(snippets)

    return issue_map

if __name__ == "__main__":
    pass
