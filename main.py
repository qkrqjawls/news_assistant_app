import os
from datetime import datetime, timezone, timedelta
import mysql.connector
from flask import Flask, request, jsonify
import numpy as np
import io
import torch
import torch.nn.functional as F

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
# --- 유틸리티 함수 (이전과 동일하거나 최적화 반영) ---
def predict_date(related_articles: list):
    dates = sorted(a['pub_date'] for a in related_articles)
    n = len(dates)
    if n == 0:
        return None
    # 날짜 데이터가 datetime 객체라고 가정
    return dates[n//4] # 25% 지점의 날짜 반환

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

# GPU 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 전역 상수 정의 (반복적인 객체 생성 방지)
FOUR_HOURS = timedelta(hours=4)

# GPU 기반 코사인 유사도 함수 (루프 밖으로 이동)
def my_cosine_similarity_torch(vec1: torch.Tensor, vec2: torch.Tensor):
    if vec1 is None or vec2 is None or vec1.shape != vec2.shape:
        # 두 벡터 중 하나라도 None이거나 형태가 다르면 -1.0 반환
        return -1.0
    # PyTorch의 F.cosine_similarity는 (N, dim) 형태를 기대하므로 reshape 필요 없음
    # 1D 벡터를 그대로 넣고 dim=0으로 합산
    return F.cosine_similarity(vec1, vec2, dim=0).item()


@app.route('/save-issues', methods=['POST'])
def save_issues_to_db():
    conn = get_db_connection()
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
    cols = ['article_id', 'title', 'content', 'pub_date']
    articles = [dict(zip(cols, row)) for row in cursor.fetchall()]

    if not articles:
        cursor.close()
        conn.close()
        return jsonify({"status":"no_new_articles"}), 200

    # 클러스터링 및 요약 (이 부분은 내부적으로 GPU를 활용해야 함)
    clustered_data, group_rep_vec = cluster_items(articles, dist_thresh=DISTANCE_THRESHOLD)
    issue_summary = summarize_and_save(clustered_data)
    date_pred = [predict_date(i) for i in clustered_data]

    issues = [{
        'date' : i[2],
        'issue_name' : i[1]['issue_name'],
        'summary' : i[1]['issue_summary'],
        'related_news_list' : [x['article_id'] for x in i[0]],
        'sentence_embedding' : i[3] # group_rep_vec (numpy array)
    } for i in zip(clustered_data, issue_summary, date_pred, group_rep_vec)]

    dates = [i['date'] for i in issues]
    min_date = min(dates) - FOUR_HOURS
    max_date = max(dates) + FOUR_HOURS

    print("issuing of new articles is done")

    # 기존 이슈 로딩 및 GPU로 임베딩 이동
    cursor.execute("""SELECT id, sentence_embedding, related_news_list, `date` FROM issues WHERE `date` > %s AND `date` < %s;""", min_date, max_date)
    existing_issues_raw = cursor.fetchall()

    existing_issues_gpu = []
    if existing_issues_raw:
        for i_id, vec_blob, related_list_str, pub_date in existing_issues_raw:
            # BLOB을 numpy 배열로 로드
            np_array = load_ndarray(vec_blob)
            if np_array is not None:
                # numpy 배열을 torch 텐서로 변환하고 GPU로 이동
                gpu_tensor = torch.from_numpy(np_array).to(device)
                existing_issues_gpu.append((i_id, gpu_tensor, related_list_str.split(), pub_date)) # related_list_str.split()

    # 신규 이슈와 기존 이슈 병합 로직
    for idx, new_issue in enumerate(issues):
        if not new_issue: # 이미 병합된 이슈는 스킵
            continue

        # 현재 신규 이슈의 임베딩을 GPU로 이동
        current_issue_emb_gpu = torch.from_numpy(new_issue['sentence_embedding']).to(device)

        # 유사성 계산 및 최대 유사 이슈 찾기
        # 날짜 범위 조건 추가 (FOUR_HOURS 사용)
        potential_matches = []
        for i_id, existing_emb_gpu, related_ids_list, issue_date in existing_issues_gpu:
            if abs(issue_date - new_issue['date']) < FOUR_HOURS:
                sim_score = my_cosine_similarity_torch(current_issue_emb_gpu, existing_emb_gpu)
                potential_matches.append((i_id, sim_score, related_ids_list, existing_emb_gpu, issue_date))

        sim = None
        if potential_matches:
            sim = max(potential_matches, key=lambda x: x[1])

        if sim and sim[1] > ISSUE_MERGING_BOUND: # sim[1]은 유사도 점수
            print(f"DEBUG: 유사한 이슈 발견. 기존 이슈 ID: {sim[0]}, 유사도: {sim[1]:.4f}")
            """유사한 이슈 발견 시 처리 -> 병합된 군집에 대한 새로운 요약 생성, 기존 id에 덮어써서 저장."""
            
            # 모든 병합될 article_id를 모읍니다. (기존 + 신규)
            all_merged_article_ids = list(set(sim[2] + new_issue['related_news_list'])) # sim[2]는 기존 이슈의 related_news_list
            
            merged_group_data = []
            if all_merged_article_ids:
                # 한 번의 쿼리로 모든 관련 기사 정보 가져오기 (IN 연산자 활용)
                placeholders = ', '.join(['%s'] * len(all_merged_article_ids))
                cursor.execute(f"""
                    SELECT article_id, title, content
                    FROM news_articles
                    WHERE article_id IN ({placeholders});
                """, tuple(all_merged_article_ids))
                cols = ['article_id', 'title', 'content']
                merged_group_data = [dict(zip(cols, row)) for row in cursor.fetchall()]

            # 병합된 기사 그룹에 대한 새로운 요약 생성
            got_list = summarize_and_save([merged_group_data]) # [merged_group_data]는 cluster_items의 출력 형태를 맞춤
            
            if not got_list or 'issue_name' not in got_list[0]:
                print(f"WARNING: 병합된 이슈 ID {sim[0]}에 대한 요약 생성 실패. 스킵.")
                continue # 또는 fallback 처리

            got = got_list[0]

            # 임베딩 평균 계산 (GPU 텐서로)
            len_a = len(sim[2]) # 기존 이슈의 기사 수
            len_b = len(new_issue['related_news_list']) # 신규 이슈의 기사 수
            
            if len_a + len_b == 0:
                print(f"WARNING: 병합할 기사가 없어 새 임베딩 계산 불가. 기존 이슈 ID: {sim[0]}. 스킵.")
                continue # 또는 예외 처리

            # GPU 텐서로 변환된 길이를 사용하여 임베딩 평균 계산
            len_a_tensor = torch.tensor(len_a, dtype=torch.float32).to(device)
            len_b_tensor = torch.tensor(len_b, dtype=torch.float32).to(device)
            
            # new_embedding_np는 numpy로 변환 후 DB에 저장
            new_embedding = (sim[3] * len_a_tensor + current_issue_emb_gpu * len_b_tensor) / (len_a_tensor + len_b_tensor)
            new_embedding_np = new_embedding.cpu().numpy() # DB 저장을 위해 다시 numpy로

            # 기존 이슈 업데이트
            cursor.execute("""UPDATE issues SET
                                issue_name        = %s,
                                summary           = %s,
                                related_news_list = %s,
                                sentence_embedding= %s
                                WHERE id = %s;""",
                            (got['issue_name'], got['issue_summary'], " ".join(all_merged_article_ids), arr_to_blob(new_embedding_np), sim[0]))
            
            # 병합된 신규 이슈는 최종 삽입 목록에서 제외
            issues[idx] = None
    print("done merging")

    # 새로운 이슈 삽입
    for issue in issues:
        if issue:
            # issue['sentence_embedding']은 이미 numpy 배열임
            cursor.execute("""INSERT IGNORE INTO issues (date, issue_name, summary, related_news_list, sentence_embedding) VALUES (%s, %s, %s, %s, %s)""",
                           (issue['date'], issue['issue_name'], issue['summary'], " ".join(issue['related_news_list']), arr_to_blob(issue['sentence_embedding'])))

    # 6) 커스텀 값 미사용 시에만 kv_int_store 갱신
    if update_kv:
        cursor.execute("""
            REPLACE INTO kv_int_store (name, my_val)
            VALUES (%s, %s);
        """, ('proceeded', full_count))

    # 삽입 또는 업데이트된 이슈 수 계산 (None이 아닌 이슈만 세기)
    final_issue_count = sum(1 for issue in issues if issue is not None)
    print(f"all done : {final_issue_count} issues are inserted or edited")

    conn.commit()

    cursor.close()
    conn.close()
    return jsonify({"status": "success",
                    "used_range": [proceeded, full_count],
                    "processed_issues_count": final_issue_count}), 200



@app.route('/')
def home():
    return "Secured API server is running!"