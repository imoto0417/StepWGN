# MySQLdbのインポート
import MySQLdb
 
# データベースへの接続とカーソルの生成
connection = MySQLdb.connect(
    host='localhost',
    user='root',
    passwd='Classic2!',
    db='test_db')
cursor = connection.cursor()
 
# ここに実行したいコードを入力します
cursor.execute("SHOW databases;")
for row in cursor:
    print (row)

# 保存を実行
connection.commit()
 
# 接続を閉じる
connection.close()