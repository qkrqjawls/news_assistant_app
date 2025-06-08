import os
from datetime import datetime, timezone
import mysql.connector
from flask import Flask, request, jsonify

app = Flask(__name__)

# 환경변수로부터 설정 읽기 (Cloud Run 배포 시 환경변수로 세팅)
DB_USER     = os.environ.get("DB_USER", "appuser")
DB_PASS     = os.environ.get("DB_PASS", "secure_app_password")
DB_NAME     = os.environ.get("DB_NAME", "myappdb")
DB_SOCKET   = os.environ.get("DB_SOCKET")   # ex) "/cloudsql/project:region:instance"


import sys
import traceback

def get_db_connection():
    try:
        if DB_SOCKET:
            return mysql.connector.connect(
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                unix_socket=DB_SOCKET,
            )
        else:
            return mysql.connector.connect(
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                host="127.0.0.1",
                port=3306
            )
    except mysql.connector.Error as err:
        print("(!) DB 연결 실패", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

@app.route("/test-db")
def test_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT NOW()")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return jsonify({"db_time": str(result[0])})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    
from get_news import fetch_recent_kr_news, serialize, parse_datetime

@app.route('/save-news', methods=['GET'])
def save_news_to_db():
    
    called_utc = datetime.now(timezone.utc)

    data = fetch_recent_kr_news(minutes=30+15, now_utc=called_utc)

    conn = get_db_connection()
    cursor = conn.cursor()

    for article in data:
        try:
            cursor.execute("""
                INSERT IGNORE INTO news_articles (
                    article_id, title, link, creator, description, content,
                    pub_date, pub_date_tz, image_url, video_url,
                    source_id, source_name, source_priority,
                    source_url, source_icon, language,
                    country, category, duplicate
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
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
            conn.commit()
            print(f"(✓) Inserted article_id={article['article_id']}")
        except Exception as e:
            print("(!) Insert failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    
    cursor.close()
    conn.close()
    return jsonify({"status": "success"}), 200

@app.route('/save_issues', methods=['POST'])
def save_issues_to_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.close()
    conn.close()
    return jsonify({"status": "success"}), 200

@app.route('/')
def home():
    return "Secured API server is running!"
