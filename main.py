from flask import Flask, jsonify, request
from google.cloud import storage
import json

app = Flask(__name__)

BUCKET_NAME = "news_assistant_main"

@app.route('/read-message', methods=['GET'])
def read_gcs_json():
    try:
        # GCS 클라이언트
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("testdata.json")

        # JSON 데이터 읽기
        content = blob.download_as_text()
        data = json.loads(content)

        # my_message 키 반환
        message = data.get("my_message", "Key 'my_message' not found")
        return jsonify({"my_message": message})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/write-test', methods=['POST'])
def write_gcs_json():
    try:
        req_data = request.get_json(force=True)
        message = req_data.get("my_message")

        # 작성할 데이터
        payload = {
            "status": "success",
            "my_message" : message
        }

        # GCS 클라이언트
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("output.json")

        # JSON 문자열로 변환해서 업로드
        blob.upload_from_string(
            data=json.dumps(payload),
            content_type='application/json'
        )

        return jsonify({"message": f"'output.json' written to GCS bucket '{BUCKET_NAME}'"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return "GCS JSON Reader is running!"
