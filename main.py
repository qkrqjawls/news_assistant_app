import os
from datetime import datetime, timezone, timedelta
import mysql.connector
from flask import Flask, request, jsonify
import numpy as np
import io
import torch
import torch.nn.functional as F
from news_issuing.embedding_and_cluster import cluster_items
from news_issuing.summarize_issues import summarize_and_save
import json

app = Flask(__name__)

# 환경변수로부터 설정 읽기 (Cloud Run 배포 시 환경변수로 세팅)
DB_USER     = os.environ.get("DB_USER", "appuser")
DB_PASS     = os.environ.get("DB_PASS", "secure_app_password")
DB_NAME     = os.environ.get("DB_NAME", "myappdb")
DB_SOCKET   = os.environ.get("DB_SOCKET")   # ex) "/cloudsql/project:region:instance"
ISSUE_MERGING_BOUND = float(os.environ.get("ISSUE_MERGING_BOUND", 0.8))
DISTANCE_THRESHOLD = float(os.environ.get("DISTANCE_THRESHOLD", 0.6))


import sys
import traceback

def get_db_connection():
    try:
        if DB_SOCKET:
            return mysql.connector.connect(
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                unix_socket=DB_SOCKET,
            )
        else:
            return mysql.connector.connect(
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                host="127.0.0.1",
                port=3306
            )
    except mysql.connector.Error as err:
        print("(!) DB 연결 실패", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

@app.route("/test-db")
def test_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT NOW()")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return jsonify({"db_time": str(result[0])})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    
from get_news import fetch_recent_kr_news, serialize, parse_datetime

@app.route('/save-news', methods=['POST'])
def save_news_to_db():
    """
    JSON body: { "time": "...", "minutes": xx}
    """
    Q = request.get_json(silent=True) or {}

    # time 파싱 (실패 시 None → 마지막에 현재 UTC로 대체)
    try:
        called_utc = parse_datetime(Q.get("time"))
    except Exception:
        called_utc = None
    called_utc = called_utc or datetime.now(timezone.utc)

    data = fetch_recent_kr_news(minutes=int(Q.get("minutes", 45)), now_utc=called_utc)

    conn = get_db_connection()
    cursor = conn.cursor()

    for article in data:
        try:
            cursor.execute("""
                INSERT IGNORE INTO news_articles (
                    article_id, title, link, creator, description, content,
                    pub_date, pub_date_tz, image_url, video_url,
                    source_id, source_name, source_priority,
                    source_url, source_icon, language,
                    country, category, duplicate
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                article["article_id"],
                article.get("title"),
                article.get("link"),
                serialize(article.get("creator")),
                article.get("description"),
                article.get("content"),
                parse_datetime(article.get("pubDate")),
                article.get("pubDateTZ"),
                article.get("image_url"),
                article.get("video_url"),
                article.get("source_id"),
                article.get("source_name"),
                article.get("source_priority"),
                article.get("source_url"),
                article.get("source_icon"),
                article.get("language"),
                serialize(article.get("country")),
                serialize(article.get("category")),
                article.get("duplicate"),
            ))
            # print(f"(✓) Inserted article_id={article['article_id']}")
        except Exception as e:
            print("(!) Insert failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    
    conn.commit()

    print("Inserted %d articles" % len(data))

    cursor.close()
    conn.close()
    return jsonify({"status": "success", "number_of_articles" : len(data)}), 200


FOUR_HOURS = timedelta(hours=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_date(articles):
    dates = sorted(a['pub_date'] for a in articles)
    return dates[len(dates)//4] if dates else None


def arr_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def load_ndarray(blob: bytes) -> np.ndarray:
    if not blob:
        return None
    buf = io.BytesIO(blob)
    return np.load(buf, allow_pickle=False)


@app.route('/save-issues', methods=['POST'])
def save_issues_to_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        conn.start_transaction()

        # 1) 인풋 파싱
        params = request.get_json(silent=True) or {}
        start_id = params.get('start_id')
        if start_id is not None:
            proceeded = int(start_id)
            update_kv = False
        else:
            cursor.execute("SELECT my_val FROM kv_int_store WHERE name=%s", ('proceeded',))
            row = cursor.fetchone()
            proceeded = row[0] or 0
            update_kv = True

        end_id = params.get('end_id')
        if end_id is not None:
            full_count = int(end_id)
            update_kv = False
        else:
            cursor.execute("SELECT MAX(id) FROM news_articles;")
            full_count = cursor.fetchone()[0] or 0

        # 유효성 검사
        if proceeded > full_count or proceeded < 0 or full_count < 0:
            return jsonify({"error": "Invalid ID range"}), 400

        # 2) 새 기사 조회
        cursor.execute(
            """
            SELECT article_id, title, content, pub_date
              FROM news_articles
             WHERE id > %s AND id <= %s
            """, (proceeded, full_count)
        )
        cols = ['article_id', 'title', 'content', 'pub_date']
        articles = [dict(zip(cols, r)) for r in cursor.fetchall()]
        if not articles:
            return jsonify({"status": "no_new_articles"}), 200

        # 3) 클러스터링 + 요약
        clustered_data, group_reprs = cluster_items(articles, dist_thresh=DISTANCE_THRESHOLD)
        summaries = summarize_and_save(clustered_data)
        issues = []
        for group, summary, _date, repr_vec in zip(clustered_data, summaries, map(predict_date, clustered_data), group_reprs):
            issues.append({
                'date': _date,
                'issue_name': summary['issue_name'],
                'summary': summary['issue_summary'],
                'related_news_list': [a['article_id'] for a in group],
                'sentence_embedding': repr_vec
            })

        # 4) 날짜 필터
        valid_dates = [i['date'] for i in issues if i['date']]
        min_date = min(valid_dates) - FOUR_HOURS
        max_date = max(valid_dates) + FOUR_HOURS

        # 5) 기존 이슈 로드
        cursor.execute(
            """
            SELECT id, sentence_embedding, related_news_list, `date`
              FROM issues
             WHERE `date` > %s AND `date` < %s
            """, (min_date, max_date)
        )
        existing = []
        for eid, blob, rel_str, pub_date in cursor.fetchall():
            arr = load_ndarray(blob)
            if arr is not None:
                existing.append((eid, torch.from_numpy(arr).to(device), json.loads(rel_str), pub_date))

        # 6) 배치 유사도 & 병합
        new_tensors = [torch.from_numpy(i['sentence_embedding']).to(device) for i in issues]
        if existing and new_tensors:
            exist_tensors, exist_meta = zip(*[ (emb, (eid, rel, dt)) for eid, emb, rel, dt in existing ])
            new_embs = torch.stack(new_tensors)
            exist_embs = torch.stack(exist_tensors)
            sims = F.cosine_similarity(new_embs.unsqueeze(1), exist_embs.unsqueeze(0), dim=-1)
            max_scores, max_idxs = sims.max(dim=1)

            for idx, issue in enumerate(issues):
                score = max_scores[idx].item()
                best = max_idxs[idx].item()
                eid, old_rel, old_date = exist_meta[best]
                if score > ISSUE_MERGING_BOUND and abs(old_date - issue['date']) < FOUR_HOURS:
                    # 병합 대상
                    merged_ids = list(set(old_rel + issue['related_news_list']))
                    # 병합 그룹 기사 가져오기
                    ph = ','.join(['%s'] * len(merged_ids))
                    cursor.execute(f"SELECT article_id,title,content FROM news_articles WHERE article_id IN ({ph})", tuple(merged_ids))
                    merged_group = [dict(zip(['article_id','title','content'], r)) for r in cursor.fetchall()]
                    summary = summarize_and_save([merged_group])[0]
                    # 임베딩 평균
                    old_emb = exist_tensors[best]
                    new_emb = new_tensors[idx]
                    na, nb = len(old_rel), len(issue['related_news_list'])
                    merged_emb = (old_emb*na + new_emb*nb) / (na+nb)
                    # DB 업데이트
                    cursor.execute(
                        """
                        UPDATE issues SET
                          issue_name=%s, summary=%s,
                          related_news_list=%s, sentence_embedding=%s
                         WHERE id=%s
                        """,
                        (
                            summary['issue_name'], summary['issue_summary'],
                            json.dumps(merged_ids), arr_to_blob(merged_emb.cpu().numpy()), eid
                        )
                    )
                    issues[idx] = None

        # 7) 신규 이슈 삽입
        for issue in issues:
            if issue:
                cursor.execute(
                    """
                    INSERT IGNORE INTO issues
                      (date, issue_name, summary, related_news_list, sentence_embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        issue['date'], issue['issue_name'], issue['summary'],
                        json.dumps(issue['related_news_list']), arr_to_blob(issue['sentence_embedding'])
                    )
                )

        # 8) KV 갱신
        if update_kv:
            cursor.execute(
                "REPLACE INTO kv_int_store (name,my_val) VALUES (%s,%s)",
                ('proceeded', full_count)
            )

        conn.commit()
        return jsonify({
            'status': 'success',
            'used_range': [proceeded, full_count],
            'processed_issues_count': sum(1 for i in issues if i)
        }), 200

    except Exception as e:
        conn.rollback()
        print("ERROR in /save-issues:", e, flush=True)
        return jsonify({'error': str(e)}), 500

    finally:
        cursor.close()
        conn.close()

