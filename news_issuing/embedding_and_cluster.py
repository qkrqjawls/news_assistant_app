import os

os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers'
os.environ['HF_HOME'] = '/tmp/huggingface'

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from news_issuing.utils import preprocess_text
import numpy as np
import torch

# GPU 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEBUG: Using device for SentenceTransformer: {device}")

def get_sentence_transformer_model(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """
    SentenceTransformer 모델을 로드하고 GPU를 사용하도록 설정합니다.
    모델은 한 번만 로드되도록 캐싱할 수 있습니다.
    """
    if not hasattr(get_sentence_transformer_model, '_model'):
        print(f"INFO: Loading SentenceTransformer model '{model_name}' to {device}...")
        get_sentence_transformer_model._model = SentenceTransformer(model_name, device=device)
    return get_sentence_transformer_model._model

def compute_embeddings(items: list) -> np.ndarray: # 이제 numpy.ndarray를 반환하도록 타입을 명시
    """
    SBERT 모델로 각 뉴스 기사(제목+본문)의 임베딩을 계산합니다.
    모델은 GPU로 로드되며, 임베딩 계산도 GPU를 활용합니다.
    - items: dict 리스트 (각 dict에 'title'과 'content' 필드가 있다고 가정)
    반환: numpy.ndarray 형태의 임베딩 배열
    """
    model = get_sentence_transformer_model() # 캐시된 모델 사용
    
    texts = [preprocess_text(x.get("title", ""), x.get("content", "")) for x in items]
    
    # 임베딩 계산 시 convert_to_tensor=True로 설정하여 PyTorch Tensor로 직접 받습니다.
    # 그러나 sklearn의 AgglomerativeClustering은 numpy 배열을 기대하므로, 
    # 여기서는 최종적으로 numpy 배열로 변환하여 반환하는 것이 클러스터링 함수와 호환성을 유지합니다.
    # 모델 자체는 GPU를 사용하므로, encode 내부적으로는 GPU에서 계산이 이루어집니다.
    embeddings_tensor = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    # sklearn 클러스터링을 위해 CPU로 이동 후 numpy 배열로 변환
    embeddings_numpy = embeddings_tensor.cpu().numpy()
    
    return embeddings_numpy

def cluster_items(items: list, dist_thresh=0.6):
    """
    뉴스 기사들을 클러스터링합니다.
    - items: 뉴스 dict 리스트. 각 dict에 최소한 'article_id', 'title', 'content'가 있어야 함.
    - dist_thresh: cosine 거리 임계값 (0.0: 완전 동일, 1.0: 완전 반대)
    
    반환값:
        - results: 각 클러스터의 뉴스 리스트 (list[list[dict]])
        - group_rep_vec: 각 클러스터의 대표 임베딩 벡터 (list[np.ndarray])
    """
    # 1) 임베딩 계산 (GPU 활용)
    # compute_embeddings는 이제 numpy.ndarray를 반환합니다.
    embeddings = compute_embeddings(items)

    # 2) 클러스터링 (sklearn.cluster.AgglomerativeClustering은 CPU에서 동작)
    # 'metric="cosine"'은 코사인 거리를 사용하며, 0에 가까울수록 유사하고 1에 가까울수록 비유사합니다.
    # 따라서 distance_threshold는 거리가 이 값보다 작을 때 병합됩니다.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine", # 코사인 거리를 사용
        linkage="average", # UPGMA (Average Linkage)
        distance_threshold=dist_thresh # 이 거리보다 가까우면 같은 클러스터로 간주
    )
    labels = clustering.fit_predict(embeddings)

    # 3) 군집별 결과 매핑 및 대표 벡터 계산
    n_clusters = max(labels) + 1 if labels.size > 0 else 0 # 빈 경우 처리

    results = [[] for _ in range(n_clusters)]
    group_rep_vec = []

    if n_clusters > 0: # 클러스터가 하나라도 있을 경우에만 처리
        for idx, lbl in enumerate(labels):
            results[lbl].append(items[idx])

        for lbl in range(n_clusters):
            indices = [i for i, l in enumerate(labels) if l == lbl]
            
            # 클러스터링 결과를 NumPy 배열로 받은 후, 평균 계산을 위해 다시 NumPy 배열 사용
            vecs = embeddings[indices] 
            
            # 평균 계산은 NumPy로 충분하며, 이 결과가 다시 DB에 저장될 NumPy 배열이 됩니다.
            centroid = np.mean(vecs, axis=0) 
            group_rep_vec.append(centroid)

    return results, group_rep_vec