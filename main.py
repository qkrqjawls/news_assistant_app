import os
import datetime
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
    
@app.route('/')
def home():
    return "Secured API server is running!"
