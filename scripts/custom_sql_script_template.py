# For sql queries that are not standard, so I would keep on changing it.

import sqlite3

conn = sqlite3.connect('database.db')

c = conn.cursor()

# c.execute("ALTER TABLE CSX ADD PRIMARY KEY (day,)")
# c.execute("DROP TABLE CSX")

conn.commit()
conn.close()