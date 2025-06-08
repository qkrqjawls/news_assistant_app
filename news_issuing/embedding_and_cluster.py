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
      embeddings (Nxd numpy array)
    """
    model = SentenceTransformer(model_name)
    texts = [preprocess_text(x.get("title", ""), x.get("body", "")) for x in items]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def cluster_and_save(input_folder: str, output_path: str, dist_thresh=0.6):
    """
    1) newsdata/ 폴더에서 모든 .json 파일 로드
    2) compute_embeddings()로 임베딩 계산
    3) AgglomerativeClustering 수행
       - metric="cosine", linkage="average", distance_threshold=dist_thresh
    4) 각 기사 id와 cluster_id만 포함한 결과 리스트 생성
    5) 결과를 JSON으로 output_path에 저장
    """
    # 1) JSON 로드
    items = load_all_json_from_folder(input_folder)

    # 2) 임베딩 계산
    embeddings = compute_embeddings(items)

    # 3) 계층적 클러스터링
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=dist_thresh
    )
    labels = clustering.fit_predict(embeddings)

    # 4) 결과 리스트 생성
    results = []
    for item, lbl in zip(items, labels):
        article_id = item.get("id")
        results.append({"id": article_id, "cluster_id": int(lbl)})

    # 5) JSON 저장
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"▶ 클러스터링 완료: '{output_path}' 생성 ({len(results)}개 항목, distance_threshold={dist_thresh})")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="뉴스 JSON을 임베딩 후 클러스터링하여 id-클러스터 매핑 JSON으로 저장"
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