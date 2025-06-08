# summarize_issues.py

import json
from collections import defaultdict
from news_issuing.utils import summarize_with_openai

def load_clustered(path: str):
    return json.load(open(path, "r", encoding="utf-8"))

def group_by_cluster(items):
    groups = defaultdict(list)
    for it in items:
        groups[it.get("cluster_id", -1)].append(it)
    return groups

def summarize_and_save(clustered_data: list):
    groups = clustered_data

    issue_map = [dict() for i in range(len(clustered_data))]
    for cid, lst in enumerate(groups):
        snippets = [
            it.get("title","") + " " + it.get("content","")[:200]
            for it in lst
        ]
        issue_map[cid] = summarize_with_openai(snippets)
        # print(f"✔ cluster {cid} summarized → {issue_map[cid]['issue_name']}")

    return issue_map

if __name__ == "__main__":
    pass
