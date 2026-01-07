import pymysql
import os

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")
db_ssl_ca = os.getenv("DB_SSL_CA")

conn = pymysql.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    database=db_name,
    port=3306,
    ssl={"ca": db_ssl_ca},
    cursorclass=pymysql.cursors.DictCursor,  # 결과를 dict로 받기
    charset="utf8mb4",
    autocommit=True,
)

try:
    with conn.cursor() as cursor:
        # 1. 테이블 목록 확인
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        print("테이블 목록:")
        for t in tables:
            print(t)

        cursor.execute("SELECT * from users;")
        users = cursor.fetchall()
        print("\nusers 테이블 데이터:")
        for user in users:
            print(user)
            

    print("\n연결 성공!")

finally:
    conn.close()
