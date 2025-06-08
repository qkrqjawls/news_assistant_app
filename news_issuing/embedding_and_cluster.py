from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from utils import preprocess_text

def compute_embeddings(items, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """
    SBERT 모델로 각 뉴스 기사(제목+본문)의 임베딩을 계산합니다.
    - items: dict 리스트, 각 dict에 'title'과 'body' 필드가 있다고 가정
    - model_name: HuggingFace 문장 임베딩 모델 이름
    반환:
      embeddings (Nxd numpy array)
    """
    model = SentenceTransformer(model_name)
    texts = [preprocess_text(item.get("title", ""), item.get("body", "")) for item in items]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def cluster_items(items: list, dist_thresh: float = 0.6) -> list:
    """
    입력된 뉴스 항목 리스트를 임베딩 후 계층적 군집화하여,
    각 항목의 'id'와 'cluster_id'만 포함된 결과 리스트를 반환합니다.

    - items: list of dict, 각 dict에 최소 'id', 'title', 'body' 필드 필요
    - dist_thresh: AgglomerativeClustering distance_threshold 값

    반환:
      list of dict, 각 dict에 'id'와 'cluster_id'
    """
    embeddings = compute_embeddings(items)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=dist_thresh
    )
    labels = clustering.fit_predict(embeddings)
    return [{"id": item.get("id"), "cluster_id": int(label)} for item, label in zip(items, labels)]
