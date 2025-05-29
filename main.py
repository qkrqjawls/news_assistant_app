from flask import Flask, jsonify
from google.cloud import storage
import json

app = Flask(__name__)

BUCKET_NAME = "news_assistant_main"
OBJECT_NAME = "testdata.json"

@app.route('/read-message', methods=['GET'])
def read_gcs_json():
    try:
        # GCS 클라이언트
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(OBJECT_NAME)

        # JSON 데이터 읽기
        content = blob.download_as_text()
        data = json.loads(content)

        # my_message 키 반환
        message = data.get("my_message", "Key 'my_message' not found")
        return jsonify({"my_message": message})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "GCS JSON Reader is running!"
