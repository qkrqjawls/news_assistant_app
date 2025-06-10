import os
from datetime import datetime, timezone, timedelta
import mysql.connector
from flask import Flask, request, jsonify
import numpy as np
import io
import torch
import torch.nn.functional as F
import sys
import traceback
import time
import logging # logging 모듈 사용 권장

# Faiss 또는 Annoy 같은 ANN 라이브러리 임포트 (설치 필요: pip install faiss-cpu 또는 pip install faiss-gpu)
# 여기서는 Faiss 예시
try:
    import faiss
    print("Faiss successfully imported.", flush=True)
except ImportError:
    print("Faiss not found. Please install faiss-cpu (pip install faiss-cpu) for ANN search.", file=sys.stderr, flush=True)
    faiss = None # Faiss가 없으면 관련 기능 비활성화 또는 경고

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
# 디버깅 시 logging.DEBUG 로 변경

# 환경변수로부터 설정 읽기
DB_USER     = os.environ.get("DB_USER", "appuser")
DB_PASS     = os.environ.get("DB_PASS", "secure_app_password")
DB_NAME     = os.environ.get("DB_NAME", "myappdb")
DB_SOCKET   = os.environ.get("DB_SOCKET")
ISSUE_MERGING_BOUND = float(os.environ.get("ISSUE_MERGING_BOUND", 0.8))
# CLUSTERING_DISTANCE_THRESHOLD는 이제 사용되지 않음 (로직 변경)

# GPU 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device for SentenceTransformer: {device}")

# 전역 상수 정의
FOUR_HOURS = timedelta(hours=4)

# 유틸리티 함수 (이전과 동일)
def get_db_connection():
    try:
        if DB_SOCKET:
            logging.info("Attempting to connect to DB via unix_socket")
            return mysql.connector.connect(
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                unix_socket=DB_SOCKET,
            )
        else:
            logging.info("Attempting to connect to DB via host:port")
            return mysql.connector.connect(
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                host="127.0.0.1",
                port=3306
            )
    except mysql.connector.Error as err:
        logging.error(f"(!) DB 연결 실패: {err}")
        traceback.print_exc()
        raise

def arr_to_blob(arr : np.ndarray):
    binary = io.BytesIO()
    np.save(binary, arr)
    return binary.getvalue()

def load_ndarray(blob: bytes) -> np.ndarray:
    if not blob:
        return None
    buf = io.BytesIO(blob)
    buf.seek(0)
    return np.load(buf, allow_pickle=False)

# get_news 모듈에서 import (가정)
from get_news import fetch_recent_kr_news, serialize, parse_datetime

# news_issuing 모듈에서 import (가정)
# embedding_and_cluster.py에서 compute_embeddings만 가져옴 (클러스터링은 사용X)
from news_issuing.embedding_and_cluster import compute_embeddings
# summarize_issues.py에서 summarize_and_save 가져옴
from news_issuing.summarize_issues import summarize_and_save

# Flask 라우트 및 기존 save-news (변동 없음)
@app.route("/test-db")
def test_db():
    start_time = time.time()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT NOW()")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        elapsed_time = time.time() - start_time
        logging.info(f"DB test successful in {elapsed_time:.2f} seconds")
        return jsonify({"db_time": str(result[0]), "elapsed_time": f"{elapsed_time:.2f}s"})
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"DB test failed in {elapsed_time:.2f} seconds: {e}")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    
@app.route('/save-news', methods=['POST'])
def save_news_to_db():
    start_total_time = time.time()
    Q = request.get_json(silent=True) or {}

    try:
        called_utc = parse_datetime(Q.get("time"))
    except Exception:
        called_utc = None
    called_utc = called_utc or datetime.now(timezone.utc)
    logging.info(f"Fetching news for minutes={Q.get('minutes', 45)} around {called_utc}")

    start_fetch_time = time.time()
    data = fetch_recent_kr_news(minutes=int(Q.get("minutes", 45)), now_utc=called_utc)
    fetch_elapsed = time.time() - start_fetch_time
    logging.info(f"Fetched {len(data)} articles in {fetch_elapsed:.2f} seconds")

    conn = get_db_connection()
    cursor = conn.cursor()

    articles_inserted = 0
    article_data_to_insert = []
    for article in data:
        article_data_to_insert.append((
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
    
    if article_data_to_insert:
        start_insert_time = time.time()
        try:
            # 배치 삽입
            cursor.executemany("""
                INSERT IGNORE INTO news_articles (
                    article_id, title, link, creator, description, content,
                    pub_date, pub_date_tz, image_url, video_url,
                    source_id, source_name, source_priority,
                    source_url, source_icon, language,
                    country, category, duplicate
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, article_data_to_insert)
            articles_inserted = cursor.rowcount # 실제 삽입된 행 수
            logging.info(f"Batch insert of {len(article_data_to_insert)} articles took {time.time() - start_insert_time:.2f} seconds. Inserted {articles_inserted} new articles.")
        except Exception as e:
            logging.error(f"(!) Batch insert failed: {e}")
            traceback.print_exc()

    start_commit_time = time.time()
    conn.commit()
    commit_elapsed = time.time() - start_commit_time
    logging.info(f"DB commit took {commit_elapsed:.2f} seconds")

    cursor.close()
    conn.close()
    total_elapsed = time.time() - start_total_time
    logging.info(f"save-news total elapsed time: {total_elapsed:.2f} seconds")
    return jsonify({"status": "success", "number_of_articles" : len(data), "articles_inserted": articles_inserted, "total_elapsed_time": f"{total_elapsed:.2f}s"}), 200

# GPU 기반 코사인 유사도 함수 (Faiss 사용 시 직접 사용될 일은 줄어듦)
# Faiss는 내부적으로 효율적인 유사도 계산을 수행
def my_cosine_similarity_torch(vec1: torch.Tensor, vec2: torch.Tensor):
    if vec1 is None or vec2 is None or vec1.shape != vec2.shape:
        return -1.0
    return F.cosine_similarity(vec1, vec2, dim=0).item()

# predict_date (이전과 동일)
def predict_date(related_articles: list):
    dates = sorted(a['pub_date'] for a in related_articles)
    n = len(dates)
    if n == 0:
        return None
    # 날짜 데이터가 datetime 객체라고 가정
    return dates[n//4] # 25% 지점의 날짜 반환

@app.route('/save-issues', methods=['POST'])
def save_issues_to_db():
    start_total_time = time.time()
    conn = get_db_connection()
    cursor = conn.cursor()
    logging.info("Starting save-issues process (new logic)...")

    Q = request.get_json(silent=True) or {}
    
    start_id_lookup_time = time.time()
    start_id = Q.get('start_id')
    if start_id is not None:
        try:
            proceeded = int(start_id)
        except (ValueError, TypeError):
            logging.error(f"Error: start_id '{start_id}' must be an integer")
            return jsonify({"error": "start_id must be an integer"}), 400
        update_kv = False
    else:
        cursor.execute("SELECT my_val FROM kv_int_store WHERE name=%s", ('proceeded',))
        row = cursor.fetchone()
        proceeded = row[0] if row and row[0] is not None else 0
        update_kv = True

    end_id = Q.get('end_id')
    if end_id is not None:
        try:
            full_count = int(end_id)
        except (ValueError, TypeError):
            logging.error(f"Error: end_id '{end_id}' must be an integer")
            return jsonify({"error": "end_id must be an integer"}), 400
        update_kv = False
    else:
        cursor.execute("SELECT MAX(id) FROM news_articles;")
        row = cursor.fetchone()
        full_count = row[0] if row and row[0] is not None else 0
    logging.info(f"ID lookup took {time.time() - start_id_lookup_time:.2f} seconds. Processing articles from ID {proceeded} to {full_count}")

    if proceeded > full_count:
        logging.warning(f"start_id ({proceeded}) cannot be greater than end_id ({full_count}). No new articles to process.")
        return jsonify({"status":"no_new_articles"}), 200
    if proceeded < 0 or full_count < 0:
        logging.error(f"IDs must be non-negative (proceeded={proceeded}, full_count={full_count})")
        return jsonify({"error":"IDs must be non-negative"}), 400

    # 1. 새로운 기사 조회 (아직 processed_article_id가 없는 기사)
    start_article_fetch_time = time.time()
    # 이미 processed_article_id가 있는 기사는 건너뛰도록 WHERE 절 추가
    # 아니면 `kv_int_store`를 이용한 `id > %s AND id <= %s` 방식 그대로 사용
    cursor.execute("""
        SELECT id, article_id, title, content, pub_date
          FROM news_articles
         WHERE id > %s AND id <= %s; 
    """, (proceeded, full_count))
    cols = ['id', 'article_id', 'title', 'content', 'pub_date']
    new_articles = [dict(zip(cols, row)) for row in cursor.fetchall()]
    logging.info(f"Fetched {len(new_articles)} new articles in {time.time() - start_article_fetch_time:.2f} seconds.")

    if not new_articles:
        cursor.close()
        conn.close()
        logging.info("No new articles to process. Exiting save-issues.")
        return jsonify({"status":"no_new_articles"}), 200

    # 2. 기존 이슈 로딩 및 GPU로 임베딩 이동
    start_existing_issues_load_time = time.time()
    cursor.execute("""SELECT id, sentence_embedding, related_news_list, `date`, issue_name FROM issues;""")
    existing_issues_raw = cursor.fetchall()

    existing_issues_info = [] # (issue_id, tensor_embedding, related_articles_list, issue_date, issue_name)
    existing_embeddings_np = [] # Faiss를 위한 numpy 배열
    existing_issue_id_map = {} # Faiss 인덱스에 매핑될 issue_id

    if existing_issues_raw:
        for i_id, vec_blob, related_list_str, pub_date, issue_name in existing_issues_raw:
            np_array = load_ndarray(vec_blob)
            if np_array is not None:
                # GPU 텐서는 유사도 직접 계산 시 사용, Faiss는 numpy를 사용
                existing_issues_info.append((i_id, torch.from_numpy(np_array).to(device), related_list_str.split(), pub_date, issue_name))
                existing_embeddings_np.append(np_array)
                existing_issue_id_map[len(existing_embeddings_np) - 1] = i_id # Faiss 내부 인덱스 -> issue_id
    
    existing_embeddings_np = np.array(existing_embeddings_np).astype('float32') # Faiss는 float32를 기대
    logging.info(f"Loaded {len(existing_issues_info)} existing issues in {time.time() - start_existing_issues_load_time:.2f} seconds.")

    # Faiss 인덱스 생성 (기존 이슈가 있을 경우)
    faiss_index = None
    if faiss and existing_embeddings_np.shape[0] > 0:
        d = existing_embeddings_np.shape[1] # 임베딩 차원
        faiss_index = faiss.IndexFlatIP(d) # 내적(Inner Product) 기반 인덱스 (코사인 유사도에 적합)
        faiss_index.add(existing_embeddings_np)
        logging.info(f"Faiss index created with {faiss_index.ntotal} vectors.")

    # 3. 각 새로운 기사 처리
    articles_to_update_kv_processed = [] # processed_article_id를 업데이트할 기사 ID
    updated_issues_for_db = {} # {issue_id: {issue_name, summary, related_news_list, sentence_embedding, date}}
    new_issues_for_db = [] # [ {date, issue_name, summary, related_news_list, sentence_embedding} ]

    start_processing_new_articles_loop_time = time.time()
    # 배치로 임베딩 계산
    new_article_embeddings = compute_embeddings(new_articles) # (num_articles, embed_dim) numpy 배열
    new_article_embeddings_gpu = torch.from_numpy(new_article_embeddings).to(device) # GPU 텐서

    for i, new_article in enumerate(new_articles):
        new_article_emb_np = new_article_embeddings[i:i+1] # (1, embed_dim) numpy
        new_article_emb_gpu = new_article_embeddings_gpu[i] # (embed_dim,) torch tensor
        
        best_match_id = None
        best_sim_score = -1.0
        
        # 4. 유사 이슈 검색 (Faiss 활용 또는 직접 유사도 계산)
        if faiss_index and faiss_index.ntotal > 0:
            # Faiss를 사용하여 K개의 최근접 이웃 검색
            # Faiss의 Inner Product는 코사인 유사도와 직접적으로 관련 (L2 정규화된 벡터의 내적)
            k = 1 # 가장 유사한 1개만 찾음
            # D: 유사도 점수, I: 해당 벡터의 인덱스 (Faiss 인덱스 내부)
            D, I = faiss_index.search(new_article_emb_np, k) 
            
            # 날짜 필터링 및 유사도 임계값 적용
            for j in range(k):
                faiss_idx = I[0][j]
                sim_score = D[0][j]
                
                # Faiss는 L2 정규화된 벡터에 대해 코사인 유사도를 직접 반환.
                # 그러나 Faiss IndexFlatIP는 실제로는 내적 값을 반환하므로, 
                # 벡터가 L2 정규화되어 있으면 내적 == 코사인 유사도이다.
                # (Faiss는 기본적으로 L2 distances를 계산하지 않고 Inner Product를 계산하기에
                # L2 정규화가 되어있는지 확실히 하는 것이 중요하다)
                
                # 기존 이슈 정보를 찾아옴
                # existing_issues_info는 (issue_id, tensor_embedding, related_articles_list, issue_date, issue_name)
                # 이 인덱스는 faiss_idx와 동일하지 않을 수 있다.
                # -> existing_issue_id_map을 사용하여 매핑해야 함.
                if faiss_idx != -1: # 유효한 인덱스
                    matched_issue_id = existing_issue_id_map[faiss_idx]
                    
                    # existing_issues_info에서 해당 이슈 찾기 (비효율적일 수 있으나, 보통 기존 이슈 수가 너무 많지는 않다고 가정)
                    # 실제로는 existing_issues_info도 딕셔너리로 미리 만들어두는 것이 좋음.
                    matched_issue = next((issue for issue in existing_issues_info if issue[0] == matched_issue_id), None)

                    if matched_issue and abs(matched_issue[3] - new_article['pub_date']) < FOUR_HOURS: # matched_issue[3] = issue_date
                        if sim_score > best_sim_score:
                            best_sim_score = sim_score
                            best_match_id = matched_issue_id
                            best_match_info = matched_issue # (i_id, existing_emb_gpu, related_ids_list, issue_date, issue_name)
        else: # Faiss가 없거나 기존 이슈가 없는 경우, 직접 유사도 계산 (비효율적이지만 폴백)
            for existing_issue_id, existing_emb_gpu, related_ids_list, issue_date, issue_name in existing_issues_info:
                if abs(issue_date - new_article['pub_date']) < FOUR_HOURS:
                    sim_score = my_cosine_similarity_torch(new_article_emb_gpu, existing_emb_gpu)
                    if sim_score > best_sim_score:
                        best_sim_score = sim_score
                        best_match_id = existing_issue_id
                        best_match_info = (existing_issue_id, existing_emb_gpu, related_ids_list, issue_date, issue_name)

        # 5. 병합 또는 신규 이슈 생성
        if best_match_id and best_sim_score > ISSUE_MERGING_BOUND:
            logging.debug(f"Article {new_article['article_id']} matched with existing issue {best_match_id} (sim: {best_sim_score:.4f})")
            
            # 병합 로직
            # 기존 이슈 업데이트를 위한 데이터 준비
            # best_match_info: (i_id, tensor_embedding, related_articles_list, issue_date, issue_name)
            existing_issue_id, existing_emb_gpu, existing_related_ids, existing_issue_date, existing_issue_name = best_match_info

            # 관련 기사 목록 업데이트
            all_merged_article_ids = list(set(existing_related_ids + [new_article['article_id']]))

            # 병합된 기사들의 임베딩 평균 계산
            # 현재 기사의 임베딩 (GPU)과 기존 이슈의 대표 임베딩 (GPU)을 사용하여 새로운 평균 임베딩 계산
            # 기존 이슈의 임베딩은 best_match_info[1] (GPU 텐서)
            len_existing = len(existing_related_ids)
            len_new_article = 1 # 새로운 기사 하나
            
            # 가중 평균 (기사 수에 비례)
            new_issue_embedding_gpu = (existing_emb_gpu * len_existing + new_article_emb_gpu * len_new_article) / (len_existing + len_new_article)
            new_issue_embedding_np = new_issue_embedding_gpu.cpu().numpy()

            # 병합된 기사들에 대한 요약 재실행 (필요한 경우)
            # 이 단계가 매우 느릴 수 있으므로, 최소한의 경우에만 수행하도록 최적화가 필요할 수 있음.
            # 예: 기사가 일정 수 이상 추가될 때만 요약 재실행
            # 여기서는 일단 모든 병합에 대해 요약 재실행으로 구현
            
            # 병합된 모든 기사의 원본 텍스트를 다시 가져와야 함 (요약 모델 입력용)
            merged_articles_full_data = []
            if all_merged_article_ids:
                placeholders = ', '.join(['%s'] * len(all_merged_article_ids))
                cursor.execute(f"""
                    SELECT article_id, title, content
                    FROM news_articles
                    WHERE article_id IN ({placeholders});
                """, tuple(all_merged_article_ids))
                cols = ['article_id', 'title', 'content']
                merged_articles_full_data = [dict(zip(cols, row)) for row in cursor.fetchall()]
            
            new_summary_result = summarize_and_save([merged_articles_full_data])[0] # 결과는 리스트의 첫 번째 요소
            
            updated_issue_id = existing_issue_id
            updated_issues_for_db[updated_issue_id] = {
                'issue_name': new_summary_result['issue_name'],
                'summary': new_summary_result['issue_summary'],
                'related_news_list': " ".join(all_merged_article_ids),
                'sentence_embedding': arr_to_blob(new_issue_embedding_np),
                'date': predict_date(merged_articles_full_data) # 병합된 기사들의 날짜로 업데이트
            }
            # 기존 issues_info 목록도 업데이트 (다음 기사 처리 시 참조할 수 있도록)
            # Find and update the corresponding item in existing_issues_info
            for k, (i_id, emb_gpu, related_ids, i_date, i_name) in enumerate(existing_issues_info):
                if i_id == existing_issue_id:
                    existing_issues_info[k] = (
                        i_id,
                        new_issue_embedding_gpu, # GPU 텐서 업데이트
                        all_merged_article_ids, # related_ids_list 업데이트
                        predict_date(merged_articles_full_data), # date 업데이트
                        new_summary_result['issue_name'] # name 업데이트
                    )
                    # Faiss 인덱스도 업데이트 해야 하지만, Faiss IndexFlat은 업데이트를 직접 지원하지 않음.
                    # -> 전체 인덱스를 다시 생성하거나, Faiss IndexIDMap 같은 것을 고려해야 함.
                    # -> 복잡성을 줄이기 위해 여기서는 Faiss 인덱스는 재성성하지 않고, 다음번 `save-issues` 호출 시 전체 로딩하도록.
                    #    단, 병합된 이슈는 다음 기사 처리 시 기존 이슈 목록에서 제외하는 처리를 하거나,
                    #    faiss_index.remove_ids()를 사용해야 함.
                    #    여기서는 업데이트된 이슈는 새로운 기사가 들어오면 Faiss 인덱스에서 검색될 수 있도록 가정
                    break


        else:
            logging.debug(f"Article {new_article['article_id']} created new issue (best sim: {best_sim_score:.4f})")
            # 새로운 이슈 생성 로직
            # 단일 기사에 대한 요약
            single_article_summary = summarize_and_save([[new_article]])[0] # summarize_and_save는 리스트의 리스트를 기대
            new_issues_for_db.append({
                'date': new_article['pub_date'], # 단일 기사의 날짜
                'issue_name': single_article_summary['issue_name'],
                'summary': single_article_summary['issue_summary'],
                'related_news_list': " ".join([new_article['article_id']]),
                'sentence_embedding': arr_to_blob(new_article_emb_np.squeeze(axis=0)) # (embed_dim,) 형태로 변환
            })
        
        # 기사가 처리되었음을 표시 (DB 업데이트는 마지막에 배치로)
        articles_to_update_kv_processed.append(new_article['id']) # news_articles 테이블의 ID

    logging.info(f"Processing new articles loop completed in {time.time() - start_processing_new_articles_loop_time:.2f} seconds.")

    # 6. DB 반영 (배치 처리)
    # 업데이트될 이슈들 반영
    if updated_issues_for_db:
        update_statements = []
        update_params = []
        for issue_id, data in updated_issues_for_db.items():
            update_statements.append(f"""
                UPDATE issues SET
                    issue_name = %s,
                    summary = %s,
                    related_news_list = %s,
                    sentence_embedding = %s,
                    date = %s
                WHERE id = %s
            """)
            update_params.append((data['issue_name'], data['summary'], data['related_news_list'], data['sentence_embedding'], data['date'], issue_id))
        
        start_batch_update_time = time.time()
        for stmt, params in zip(update_statements, update_params): # executemany는 단일 쿼리에만 적용되므로, 여러 쿼리는 개별 실행
            cursor.execute(stmt, params)
        logging.info(f"Batch update for {len(updated_issues_for_db)} existing issues took {time.time() - start_batch_update_time:.2f} seconds.")
        
    # 새로운 이슈들 삽입
    if new_issues_for_db:
        insert_data = []
        for issue in new_issues_for_db:
            insert_data.append((issue['date'], issue['issue_name'], issue['summary'], issue['related_news_list'], issue['sentence_embedding']))
        
        start_batch_insert_time = time.time()
        cursor.executemany("""INSERT IGNORE INTO issues (date, issue_name, summary, related_news_list, sentence_embedding) VALUES (%s, %s, %s, %s, %s)""", insert_data)
        logging.info(f"Batch inserted {cursor.rowcount} new issues in {time.time() - start_batch_insert_time:.2f} seconds.")
    
    # kv_int_store 갱신 (처리된 마지막 article ID)
    if update_kv:
        start_kv_update_time = time.time()
        cursor.execute("""
            REPLACE INTO kv_int_store (name, my_val)
            VALUES (%s, %s);
        """, ('proceeded', full_count)) # 마지막으로 처리된 article id
        logging.info(f"kv_int_store updated to {full_count} in {time.time() - start_kv_update_time:.2f} seconds.")

    start_final_commit_time = time.time()
    conn.commit()
    logging.info(f"Final DB commit took {time.time() - start_final_commit_time:.2f} seconds.")

    cursor.close()
    conn.close()
    total_elapsed = time.time() - start_total_time
    logging.info(f"save-issues total elapsed time: {total_elapsed:.2f} seconds.")
    
    return jsonify({"status": "success",
                    "used_range": [proceeded, full_count],
                    "new_issues_created": len(new_issues_for_db),
                    "issues_updated": len(updated_issues_for_db),
                    "total_elapsed_time": f"{total_elapsed:.2f}s"}), 200

@app.route('/')
def home():
    logging.info("Home route accessed.")
    return "Secured API server is running!"