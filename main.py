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

def predict_date(related_articles: list):
    valid_dates = []
    for a in related_articles:
        pub_date = a.get('pub_date')
        if pub_date:
            if isinstance(pub_date, datetime):
                valid_dates.append(pub_date)
            else:
                try:
                    parsed_date = parse_datetime(pub_date)
                    if parsed_date:
                        valid_dates.append(parsed_date)
                except Exception as e:
                    logging.warning(f"Failed to parse date '{pub_date}': {e}")
    
    if not valid_dates:
        logging.warning("No valid 'pub_date' found in related articles. Returning None for date prediction.")
        return None
    
    dates = sorted(valid_dates)
    n = len(dates)
    
    # 25% 지점 유지 (원하는 경우)
    return dates[n // 4] 
    # 또는 가장 최신 날짜: return dates[-1]

# ... (상단 import 및 Flask 초기화 등 동일) ...

@app.route('/save-issues', methods=['POST'])
def save_issues_to_db():
    start_total_time = time.time()
    conn = get_db_connection()
    cursor = conn.cursor()
    logging.info("Starting save-issues process (new logic with Faiss IDMap and immediate new issue add)...")

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

    # 1. 새로운 기사 조회
    start_article_fetch_time = time.time()
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

    existing_issues_data = {} # {issue_id: (tensor_embedding, related_articles_list, issue_date, issue_name)}
    existing_embeddings_np_list = [] # Faiss를 위한 numpy 배열 리스트
    existing_issue_ids_for_faiss = [] # Faiss에 추가될 issue_id 리스트

    if existing_issues_raw:
        for i_id, vec_blob, related_list_str, pub_date, issue_name in existing_issues_raw:
            np_array = load_ndarray(vec_blob)
            if np_array is not None:
                existing_embeddings_np_list.append(np_array.astype('float32')) 
                existing_issue_ids_for_faiss.append(i_id)
                existing_issues_data[i_id] = (torch.from_numpy(np_array).to(device), related_list_str.split(), pub_date, issue_name)
    
    # Faiss 인덱스 생성 (IndexIDMap 사용)
    faiss_index = None
    if faiss and len(existing_embeddings_np_list) > 0:
        d = existing_embeddings_np_list[0].shape[0] # 임베딩 차원
        quantizer = faiss.IndexFlatIP(d) 
        faiss_index = faiss.IndexIDMap(quantizer)
        
        # 일괄 추가
        faiss_index.add_with_ids(np.array(existing_embeddings_np_list), np.array(existing_issue_ids_for_faiss))
        logging.info(f"Faiss IndexIDMap created with {faiss_index.ntotal} vectors.")
    else:
        # 기존 이슈가 없더라도, Faiss가 있으면 빈 인덱스를 생성해 둡니다.
        # 새로운 이슈가 생성될 때 즉시 추가할 수 있도록.
        if faiss and new_articles: # 새로운 기사가 있어서 임베딩 차원을 알 수 있을 때
            # compute_embeddings가 반환할 임베딩 차원을 미리 예상
            # (이것은 최적의 방법은 아니지만, 현재 흐름에서 임베딩 차원을 알기 위한 임시 방편)
            # compute_embeddings 함수의 반환값 구조에 따라 수정 필요
            # 예를 들어, compute_embeddings가 최소 1개의 임베딩을 항상 반환한다면:
            temp_emb = compute_embeddings([new_articles[0]])
            if temp_emb.shape[0] > 0:
                d = temp_emb.shape[1]
                quantizer = faiss.IndexFlatIP(d) 
                faiss_index = faiss.IndexIDMap(quantizer)
                logging.info(f"Faiss IndexIDMap initialized (empty) for dimension {d}.")
            else:
                logging.warning("Could not determine embedding dimension from new articles to initialize Faiss.")
        elif faiss: # 새로운 기사가 없는데도 Faiss를 초기화해야 한다면, 미리 정의된 임베딩 차원 사용
             # 예를 들어, EMBEDDING_DIM = 768 와 같은 상수를 정의
             # d = EMBEDDING_DIM
             logging.warning("Faiss will not be initialized as no existing issues and no new articles to determine dimension.")

    logging.info(f"Loaded {len(existing_issues_data)} existing issues in {time.time() - start_existing_issues_load_time:.2f} seconds.")

    # 3. 각 새로운 기사 처리
    articles_to_update_kv_processed = [] 
    updated_issues_for_db = {} 
    # new_issues_for_db는 이제 각 이슈를 바로 처리하므로 필요 없음

    start_processing_new_articles_loop_time = time.time()
    # 배치로 임베딩 계산 (compute_embeddings는 numpy 배열을 반환한다고 가정)
    new_article_embeddings_np = compute_embeddings(new_articles) # (num_articles, embed_dim) numpy 배열
    # 모든 임베딩이 L2 정규화되어 있는지 확인 (compute_embeddings 내부에서 처리하거나 여기서 명시적으로)
    if new_article_embeddings_np.shape[0] > 0:
        new_article_embeddings_np = F.normalize(torch.from_numpy(new_article_embeddings_np), p=2, dim=1).numpy()
    
    new_article_embeddings_gpu = torch.from_numpy(new_article_embeddings_np).to(device) # GPU 텐서

    # 이 시점에서 커밋되지 않은 트랜잭션이 있을 수 있으므로,
    # 중간에 DB 연결이 끊어지면 롤백될 수 있습니다.
    # 하지만 Cloud Run의 단일 요청 처리 특성상 큰 문제는 아닙니다.

    new_issues_count_in_this_run = 0 # 이 실행에서 생성된 새 이슈 수
    
    for i, new_article in enumerate(new_articles):
        new_article_emb_np = new_article_embeddings_np[i:i+1].astype('float32') # (1, embed_dim) numpy
        new_article_emb_gpu = new_article_embeddings_gpu[i] # (embed_dim,) torch tensor
        
        best_match_id = None
        best_sim_score = -1.0
        
        # 4. 유사 이슈 검색 (Faiss 활용)
        if faiss_index and faiss_index.ntotal > 0:
            k = 1 # 가장 유사한 1개만 찾음
            # D: 유사도 점수, I: 해당 벡터의 ID (우리의 issue_id)
            D, I = faiss_index.search(new_article_emb_np, k) 
            
            for j in range(k):
                matched_issue_id = I[0][j]
                sim_score = D[0][j]
                
                # matched_issue_id != -1는 Faiss가 유효한 매치를 찾았음을 의미
                # matched_issue_id in existing_issues_data는 Faiss가 반환한 ID가
                # 현재 메모리에 로드된 기존 이슈 데이터에 실제로 존재하는지 확인 (만약 DB에만 있고 메모리에 없는 경우 방지)
                if matched_issue_id != -1 and matched_issue_id in existing_issues_data: 
                    matched_issue_info = existing_issues_data[matched_issue_id]
                    
                    # 날짜 필터링 (4시간 이내)
                    if abs(matched_issue_info[2] - new_article['pub_date']) < FOUR_HOURS: 
                        if sim_score > best_sim_score:
                            best_sim_score = sim_score
                            best_match_id = matched_issue_id
                            best_match_info = matched_issue_info 

        # 5. 병합 또는 신규 이슈 생성
        if best_match_id and best_sim_score > ISSUE_MERGING_BOUND:
            logging.debug(f"Article {new_article['article_id']} matched with existing issue {best_match_id} (sim: {best_sim_score:.4f})")
            
            # 병합 로직
            existing_emb_gpu, existing_related_ids, existing_issue_date, existing_issue_name = best_match_info

            all_merged_article_ids = list(set(existing_related_ids + [new_article['article_id']]))

            # 병합된 기사들의 임베딩 평균 계산 (가중 평균)
            len_existing = len(existing_related_ids)
            len_new_article = 1 
            
            new_issue_embedding_gpu = (existing_emb_gpu * len_existing + new_article_emb_gpu * len_new_article) / (len_existing + len_new_article)
            new_issue_embedding_np = new_issue_embedding_gpu.cpu().numpy()

            merged_articles_full_data = []
            if all_merged_article_ids:
                placeholders = ', '.join(['%s'] * len(all_merged_article_ids))
                # 기존 DB 커넥션과 커서를 재사용
                cursor.execute(f"""
                    SELECT article_id, title, content, pub_date
                    FROM news_articles
                    WHERE article_id IN ({placeholders});
                """, tuple(all_merged_article_ids))
                cols = ['article_id', 'title', 'content', 'pub_date']
                merged_articles_full_data = [dict(zip(cols, row)) for row in cursor.fetchall()]
            
            new_summary_result = summarize_and_save([merged_articles_full_data])[0] 
            
            updated_issue_id = best_match_id
            
            # DB UPDATE 즉시 실행
            cursor.execute("""
                UPDATE issues SET
                    issue_name = %s,
                    summary = %s,
                    related_news_list = %s,
                    sentence_embedding = %s,
                    date = %s
                WHERE id = %s
            """, (new_summary_result['issue_name'], new_summary_result['issue_summary'], 
                  " ".join(all_merged_article_ids), arr_to_blob(new_issue_embedding_np), 
                  predict_date(merged_articles_full_data), int(updated_issue_id)))
            updated_issues_for_db[updated_issue_id] = True # 업데이트된 이슈 ID만 기록 (상태 정보는 필요 없음)
            
            # 기존 issues_data 딕셔너리 업데이트 (다음 기사 처리 시 참조할 수 있도록)
            existing_issues_data[updated_issue_id] = (
                new_issue_embedding_gpu, # GPU 텐서 업데이트
                all_merged_article_ids, # related_ids_list 업데이트
                predict_date(merged_articles_full_data), # date 업데이트
                new_summary_result['issue_name'] # name 업데이트
            )
            
            # Faiss IndexIDMap에서 기존 벡터 제거 후 새로운 벡터 추가
            if faiss_index:
                new_issue_embedding_np_2d = new_issue_embedding_np.reshape(1, -1)
                faiss_index.remove_ids(np.array([updated_issue_id]))
                faiss_index.add_with_ids(new_issue_embedding_np_2d.astype('float32'), np.array([updated_issue_id]))
                logging.debug(f"Faiss index updated for issue ID {updated_issue_id}. Faiss ntotal: {faiss_index.ntotal}")

        else:
            logging.debug(f"Article {new_article['article_id']} created new issue (best sim: {best_sim_score:.4f})")
            
            single_article_summary = summarize_and_save([[new_article]])[0] 
            
            # 새로운 이슈를 DB에 즉시 삽입하고 ID를 가져옴
            cursor.execute("""
                INSERT IGNORE INTO issues (date, issue_name, summary, related_news_list, sentence_embedding) 
                VALUES (%s, %s, %s, %s, %s)
            """, (new_article['pub_date'], single_article_summary['issue_name'], 
                  single_article_summary['issue_summary'], " ".join([new_article['article_id']]), 
                  arr_to_blob(new_article_emb_np.squeeze(axis=0))))
            
            # LAST_INSERT_ID()를 사용하여 방금 삽입된 행의 ID를 가져옵니다.
            # IGNORE가 있으므로, 이미 존재하는 경우 0을 반환할 수 있습니다.
            new_issue_id = cursor.lastrowid
            
            if new_issue_id and new_issue_id != 0: # 실제로 새로운 행이 삽입된 경우
                new_issues_count_in_this_run += 1
                logging.debug(f"New issue created in DB with ID: {new_issue_id}")
                
                # 생성된 새 이슈 정보를 existing_issues_data에 추가 (다음 기사 처리 시 참조)
                existing_issues_data[new_issue_id] = (
                    new_article_emb_gpu, 
                    [new_article['article_id']], 
                    new_article['pub_date'], 
                    single_article_summary['issue_name']
                )
                
                # Faiss 인덱스에 새로운 벡터 추가
                if faiss_index:
                    faiss_index.add_with_ids(new_article_emb_np.astype('float32'), np.array([new_issue_id]))
                    logging.debug(f"New issue ID {new_issue_id} added to Faiss index. Faiss ntotal: {faiss_index.ntotal}")
                elif faiss: # Faiss 인덱스가 아직 초기화되지 않았을 경우, 이 시점에서 초기화 시도
                    # 단, 첫 번째 새 이슈가 이 경로로 오면 Faiss 인덱스 생성 시도를 할 수 있음.
                    # 임베딩 차원은 이미 `new_article_embeddings_np`에서 가져왔으므로 문제가 없음.
                    d = new_article_emb_np.shape[1]
                    quantizer = faiss.IndexFlatIP(d) 
                    faiss_index = faiss.IndexIDMap(quantizer)
                    faiss_index.add_with_ids(new_article_emb_np.astype('float32'), np.array([new_issue_id]))
                    logging.info(f"Faiss IndexIDMap initialized and new issue ID {new_issue_id} added. Faiss ntotal: {faiss_index.ntotal}")
                else:
                    logging.warning(f"Faiss not available, new issue ID {new_issue_id} not added to Faiss.")
            else:
                logging.warning(f"New issue for article {new_article['article_id']} was not inserted (possibly duplicate or DB error).")

        articles_to_update_kv_processed.append(new_article['id']) 

    logging.info(f"Processing new articles loop completed in {time.time() - start_processing_new_articles_loop_time:.2f} seconds.")

    # 6. DB 반영 (남은 부분 및 kv_int_store 업데이트)
    # 이미 루프 내에서 이슈 UPDATE/INSERT가 발생했으므로, 여기서 추가적인 배치 처리는 필요 없음.
    # 하지만, 트랜잭션의 안정성을 위해 커밋은 여기서 한 번만 수행합니다.

    if update_kv:
        start_kv_update_time = time.time()
        cursor.execute("""
            REPLACE INTO kv_int_store (name, my_val)
            VALUES (%s, %s);
        """, ('proceeded', full_count)) 
        logging.info(f"kv_int_store updated to {full_count} in {time.time() - start_kv_update_time:.2f} seconds.")

    start_final_commit_time = time.time()
    conn.commit() # 모든 변경사항을 한 번에 커밋
    logging.info(f"Final DB commit took {time.time() - start_final_commit_time:.2f} seconds.")

    cursor.close()
    conn.close()
    total_elapsed = time.time() - start_total_time
    logging.info(f"save-issues total elapsed time: {total_elapsed:.2f} seconds.")
    
    return jsonify({"status": "success",
                    "used_range": [proceeded, full_count],
                    "new_issues_created": new_issues_count_in_this_run, # 이 실행에서 생성된 새 이슈 수
                    "issues_updated": len(updated_issues_for_db),
                    "total_elapsed_time": f"{total_elapsed:.2f}s"}), 200

# ... (home 라우트 동일) ...@app.route('/')
def home():
    logging.info("Home route accessed.")
    return "Secured API server is running!"