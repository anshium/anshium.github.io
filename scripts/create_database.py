import sqlite3

conn = sqlite3.connect('database.db')

c = conn.cursor()

branch = input("Enter branch name: ")

c.execute('''CREATE TABLE IF NOT EXISTS CSX (day TEXT PRIMARY KEY, period_1 TEXT, period_2 TEXT, period_3 TEXT, period_4 TEXT, period_5 TEXT, period_6 TEXT)''')

conn.commit()
conn.close()