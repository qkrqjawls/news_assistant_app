from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from news_issuing.utils import preprocess_text

def compute_embeddings(items, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """
    SBERT 모델로 각 뉴스 기사(제목+본문)의 임베딩을 계산합니다.
    - items: dict 리스트 (각 dict에 'title'과 'content' 필드가 있다고 가정)
    """
    model = SentenceTransformer(model_name)
    texts = [preprocess_text(x.get("title", ""), x.get("content", "")) for x in items]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def cluster_items(items: list, dist_thresh=0.6) -> list:
    """
    - items: 뉴스 dict 리스트. 각 dict에 최소한 'article_id', 'title', 'content'가 있어야 함.
    - dist_thresh: cosine 거리 임계값
    반환값: [[article : dict, ...], ...] : list
    """
    # 1) 임베딩 계산
    embeddings = compute_embeddings(items)

    # 2) 계층적 클러스터링
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=dist_thresh
    )
    labels = clustering.fit_predict(embeddings)

    # 3) 결과 매핑
    results = [[] for i in range(max(labels)+1)]
    for item, lbl in zip(items, labels):
        results[int(lbl)].append(item)

    return results
