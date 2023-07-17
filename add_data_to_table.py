import sqlite3

conn = sqlite3.connect('database.db')

c = conn.cursor()

day = input("Enter Day: ")

period_1 = input("Enter Period 1: ")
period_2 = input("Enter Period 2: ")
period_3 = input("Enter Period 3: ")
print("Period 4 is Lunch")
period_5 = input("Enter Period 5: ")
period_6 = input("Enter Period 6: ")

conn.execute('''INSERT INTO CSX VALUES (?, ?,?,?,?,?,?)''', (day, period_1, period_2, period_3, "Lunch", period_5, period_6))

conn.commit()
conn.close()