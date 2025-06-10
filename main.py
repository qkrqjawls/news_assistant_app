import os
from datetime import datetime, timezone, timedelta
import mysql.connector
from flask import Flask, request, jsonify
import numpy as np
import io

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

from news_issuing.embedding_and_cluster import cluster_items
from news_issuing.summarize_issues import summarize_and_save
from sklearn.metrics.pairwise import cosine_similarity

def predict_date(related_articles: list):
    dates = sorted(a['pub_date'] for a in related_articles)
    n = len(dates)
    if n == 0:
        return None

    return dates[n//4]

def arr_to_blob(arr : np.ndarray):
    binary = io.BytesIO()
    np.save(binary, arr)
    binary_value = binary.getvalue()
    return binary_value

def load_ndarray(blob: bytes) -> np.ndarray:
    if not blob:
        return None
    buf = io.BytesIO(blob)
    buf.seek(0)
    return np.load(buf, allow_pickle=False)

def my_cosine_similarity(vec1 : np.ndarray, vec2 : np.ndarray): # 1D cosine sim
    if vec1 is None or vec2 is None or vec1.shape != vec2.shape:
        return -1  # 또는 0.0
    return cosine_similarity(
        vec1.reshape(1, -1),
        vec2.reshape(1, -1)
    )[0, 0]

def id_to_article(article_id : str, cursor):
    cursor.execute("""SELECT title, content FROM news_articles WHERE article_id = %s;""", (article_id,))
    row = cursor.fetchone()
    if not row:
        return {"article_id": article_id, "title": "", "content": ""}
    return {
        "article_id" : article_id,
        "title" : row[0],
        "content" : row[1]
    }


@app.route('/save-issues', methods=['POST'])
def save_issues_to_db():
    conn   = get_db_connection()
    cursor = conn.cursor()

    # 1) JSON body 파싱 (None 안전 처리)
    Q = request.get_json(silent=True) or {}

    # 2) start_id / proceeded 결정
    start_id = Q.get('start_id')
    if start_id is not None:
        try:
            proceeded = int(start_id)
        except (ValueError, TypeError):
            return jsonify({"error": "start_id must be an integer"}), 400
        update_kv = False
    else:
        cursor.execute("SELECT my_val FROM kv_int_store WHERE name=%s",
                       ('proceeded',))
        row = cursor.fetchone()
        proceeded = row[0] if row and row[0] is not None else 0
        update_kv = True

    # 3) end_id / full_count 결정
    end_id = Q.get('end_id')
    if end_id is not None:
        try:
            full_count = int(end_id)
        except (ValueError, TypeError):
            return jsonify({"error": "end_id must be an integer"}), 400
        update_kv = False
    else:
        cursor.execute("SELECT MAX(id) FROM news_articles;")
        row = cursor.fetchone()
        full_count = row[0] if row and row[0] is not None else 0

    # 4) id 범위 유효성 검사
    if proceeded > full_count:
        return jsonify({"error": "start_id cannot be greater than end_id"}), 400
    if proceeded < 0 or full_count < 0:
        return jsonify({"error":"IDs must be non-negative"}), 400

    # 5) 실제 기사 조회
    cursor.execute("""
        SELECT article_id, title, content, pub_date
          FROM news_articles
         WHERE id > %s AND id <= %s;
    """, (proceeded, full_count))
    cols     = ['article_id', 'title', 'content', 'pub_date']
    articles = [dict(zip(cols, row)) for row in cursor.fetchall()]

    if not articles:
        cursor.close()
        conn.close()
        return jsonify({"status":"no_new_articles"}), 200

    clustered_data, group_rep_vec = cluster_items(articles, dist_thresh=DISTANCE_THRESHOLD)
    issue_summary = summarize_and_save(clustered_data)
    date_pred = [predict_date(i) for i in clustered_data]

    issues = [{
        'date' : i[2],
        'issue_name' : i[1]['issue_name'],
        'summary' : i[1]['issue_summary'],
        'related_news_list' : [x['article_id'] for x in i[0]],
        'sentence_embedding' : i[3]
    } for i in zip(clustered_data, issue_summary, date_pred, group_rep_vec)]

    # print("clustering and summerization:", issues)

    cursor.execute("""SELECT id, sentence_embedding, related_news_list, `date` FROM issues;""")
    existing_issues = cursor.fetchall()
    if existing_issues:
        existing_issues = [
            (i, arr, l.split(), d)
            for i, vec, l, d in existing_issues
            if (arr := load_ndarray(vec)) is not None
        ]

        for idx, issue in enumerate(issues):
            sim = max([(i, my_cosine_similarity(issue['sentence_embedding'], vec), l, vec, d) for i,vec,l,d in existing_issues], key=lambda x:x[1])
            # sim = (issue_id, similarity_score, article_ids, embedding_vector, date)
            if sim[1] > ISSUE_MERGING_BOUND and abs(sim[4] - issue['date']) < timedelta(hours=2):
                """유사한 이슈 발견 시 처리 -> 병합된 군집에 대한 새로운 요약 생성, 기존 id에 덮어써서 저장."""
                merged_group = list(map(id_to_article, set(sim[2] + issue['related_news_list'])))
                got_list = summarize_and_save([merged_group])
                if not got_list or 'issue_name' not in got_list[0]:
                    continue  # 또는 fallback 처리
                got = got_list[0]

                len_a = len(sim[2])
                len_b = len(issue['related_news_list'])
                if len_a + len_b == 0:
                    continue  # 또는 예외 처리
                new_embedding = (sim[3]*len_a + issue['sentence_embedding']*len_b) / (len_a + len_b)

                cursor.execute("""UPDATE issues SET
                            issue_name    = %s,
                            summary           = %s,
                            related_news_list = %s,
                            sentence_embedding= %s
                            WHERE id = %s;""",
                        (got['issue_name'], got['issue_summary'], " ".join(article['article_id'] for article in merged_group), arr_to_blob(new_embedding), sim[0]))
                issues[idx] = None

    for issue in issues:
        if issue:
            cursor.execute("""INSERT IGNORE INTO issues (date, issue_name, summary, related_news_list, sentence_embedding) VALUES (%s, %s, %s, %s, %s)""",
                           (issue['date'], issue['issue_name'], issue['summary'], " ".join(issue['related_news_list']), arr_to_blob(issue['sentence_embedding'])))

    # 6) 커스텀 값 미사용 시에만 kv_int_store 갱신
    if update_kv:
        cursor.execute("""
            REPLACE INTO kv_int_store (name, my_val)
            VALUES (%s, %s);
        """, ('proceeded', full_count))

    conn.commit()

    cursor.close()
    conn.close()
    return jsonify({"status": "success",
                    "used_range": [proceeded, full_count]}), 200

@app.route('/')
def home():
    return "Secured API server is running!"
