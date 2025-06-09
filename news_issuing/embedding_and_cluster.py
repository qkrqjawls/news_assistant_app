import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers'
os.environ['HF_HOME'] = '/tmp/huggingface'

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from news_issuing.utils import preprocess_text
import numpy as np

def compute_embeddings(items, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """
    SBERT 모델로 각 뉴스 기사(제목+본문)의 임베딩을 계산합니다.
    - items: dict 리스트 (각 dict에 'title'과 'content' 필드가 있다고 가정)
    """
    model = SentenceTransformer(model_name)
    texts = [preprocess_text(x.get("title", ""), x.get("content", "")) for x in items]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def cluster_items(items: list, dist_thresh=0.6):
    """
    - items: 뉴스 dict 리스트. 각 dict에 최소한 'article_id', 'title', 'content'가 있어야 함.
    - dist_thresh: cosine 거리 임계값
    반환값:
        - results: 각 클러스터의 뉴스 리스트 (list[list[dict]])
        - group_rep_vec: 각 클러스터의 대표 임베딩 벡터 (list[np.ndarray])
    """
    # 1) 임베딩 계산
    embeddings = compute_embeddings(items)

    # 2) 클러스터링
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=dist_thresh
    )
    labels = clustering.fit_predict(embeddings)

    # 3) 군집별 결과 매핑
    n_clusters = max(labels) + 1
    results = [[] for _ in range(n_clusters)]
    group_rep_vec = []

    for idx, lbl in enumerate(labels):
        results[lbl].append(items[idx])

    for lbl in range(n_clusters):
        indices = [i for i, l in enumerate(labels) if l == lbl]
        vecs = embeddings[indices]
        centroid = np.mean(vecs, axis=0)
        group_rep_vec.append(centroid)

    return results, group_rep_vec