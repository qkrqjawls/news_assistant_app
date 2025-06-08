# generate_output.py

import json
from collections import defaultdict

def build_issues(clustered_path: str, issue_map_path: str, output_path: str):
    clustered = json.load(open(clustered_path, "r", encoding="utf-8"))
    issue_map = json.load(open(issue_map_path, "r", encoding="utf-8"))

    groups = defaultdict(list)
    for art in clustered:
        cid = art.get("cluster_id", -1)
        groups[cid].append({
            "id": art.get("id"),
            "title": art.get("title"),
            "body": art.get("body"),
            "published_at": art.get("published_at"),
            "source": art.get("source", "")
        })

    issues = []
    for cid, arts in groups.items():
        info = issue_map.get(str(cid)) or issue_map.get(int(cid), {})
        issues.append({
            "issue_id": cid,
            "issue_name": info.get("issue_name", ""),
            "issue_summary": info.get("issue_summary", ""),
            "articles": arts
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)
    print(f"▶ final issues saved to {output_path} (count: {len(issues)})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--clustered_path", default="clustered.json",
                   help="clustered.json 경로")
    p.add_argument("--issue_map_path", default="issue_map.json",
                   help="issue_map.json 경로")
    p.add_argument("--output_path", default="issues_output.json",
                   help="최종 issues_output.json 경로")
    args = p.parse_args()
    build_issues(args.clustered_path, args.issue_map_path, args.output_path)
