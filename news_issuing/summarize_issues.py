# summarize_issues.py

import json
from collections import defaultdict
from utils import summarize_with_openai

def load_clustered(path: str):
    return json.load(open(path, "r", encoding="utf-8"))

def group_by_cluster(items):
    groups = defaultdict(list)
    for it in items:
        groups[it.get("cluster_id", -1)].append(it)
    return groups

def summarize_and_save(clustered_path: str, output_path: str):
    """clustered.json → 이슈별 요약(issue_map.json) 생성"""
    items = load_clustered(clustered_path)
    groups = group_by_cluster(items)

    issue_map = {}
    for cid, lst in groups.items():
        snippets = [
            it.get("title","") + " " + it.get("body","")[:200]
            for it in lst
        ]
        issue_map[cid] = summarize_with_openai(snippets)
        print(f"✔ cluster {cid} summarized → {issue_map[cid]['issue_name']}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(issue_map, f, ensure_ascii=False, indent=2)
    print(f"▶ issue_map saved to {output_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--clustered_path", default="clustered.json",
                   help="clustered.json 경로")
    p.add_argument("--output_path", default="issue_map.json",
                   help="출력할 issue_map.json 경로")
    args = p.parse_args()
    summarize_and_save(args.clustered_path, args.output_path)
