# embedding_and_cluster.py

import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from utils import load_all_json_from_folder, preprocess_text

def compute_embeddings(items, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """
    SBERT 모델로 각 뉴스 기사(제목+본문)의 임베딩을 계산합니다.
    - items: JSON dict 리스트 (각 dict에 'title'과 'body' 필드가 있다고 가정)
    - model_name: HuggingFace 문장 임베딩 모델 이름
    반환:
      embeddings (N×d numpy array)
    """
    model = SentenceTransformer(model_name)
    texts = [preprocess_text(x.get("title", ""), x.get("body", "")) for x in items]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def cluster_and_save(input_folder: str, output_path: str, dist_thresh=0.6):
    """
    1) newsdata/ 폴더에서 모든 .json 파일 로드
    2) compute_embeddings()로 임베딩 계산
    3) AgglomerativeClustering(거리 기반 계층적 군집화) 수행
       - metric="cosine", linkage="average", distance_threshold=dist_thresh
    4) 각 기사 dict에 cluster_id 필드 추가
    5) clustered.json 형식으로 output_path 에 저장
    """
    # 1) JSON 로드
    items = load_all_json_from_folder(input_folder)

    # 2) 임베딩 계산
    embeddings = compute_embeddings(items)

    # 3) 계층적 클러스터링
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",          # scikit-learn 최신판에서 affinity → metric 으로 변경
        linkage="average",
        distance_threshold=dist_thresh
    )
    labels = clustering.fit_predict(embeddings)

    # 4) 결과에 cluster_id 추가
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    clustered = []
    for item, lbl in zip(items, labels):
        item["cluster_id"] = int(lbl)
        clustered.append(item)

    # 5) clustered.json 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clustered, f, ensure_ascii=False, indent=2)

    print(f"▶ 클러스터링 완료: '{output_path}' 생성 (distance_threshold={dist_thresh})")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="뉴스 JSON을 임베딩 후 클러스터링하여 clustered.json으로 저장"
    )
    parser.add_argument(
        "--input_folder",
        default="newsdata",
        help="원본 뉴스 JSON 파일들이 있는 폴더 경로"
    )
    parser.add_argument(
        "--output_path",
        default="clustered.json",
        help="클러스터링 결과 JSON 저장 경로"
    )
    parser.add_argument(
        "--dist_thresh",
        type=float,
        default=0.6,
        help="AgglomerativeClustering distance_threshold 값 (0~1)"
    )
    args = parser.parse_args()

    cluster_and_save(
        input_folder=args.input_folder,
        output_path=args.output_path,
        dist_thresh=args.dist_thresh
    )
