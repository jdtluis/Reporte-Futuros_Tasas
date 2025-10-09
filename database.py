import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# --- 1. Create sample DataFrame ---
df = pd.read_csv('tasas_cau_simul.csv')
df.Fecha = pd.to_datetime(df.Fecha)

# --- 2. Create SQLite connection ---
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# --- 3. Create table ---
cursor.execute('''
CREATE TABLE IF NOT EXISTS rates (
    Fecha TEXT PRIMARY KEY,
    Plazo INTEGER,
    Caucion REAL,
    Simul REAL
)
''')

# --- 4. Insert DataFrame into table ---
df['Fecha'] = df['Fecha'].astype(str)  # Convert datetime to text for SQLite

df.to_sql('rates', conn, if_exists='append', index=False)

# --- 5. Verify ---
print("\nData from database:")
print(pd.read_sql("SELECT * FROM rates", conn))

conn.close()
# %%
# Insert data
conn = sqlite3.connect('data.db')

new_date = datetime.now()
new_caucion = 0.75
new_simul = 0.85

conn.execute("""
INSERT INTO rates (date, caucion, simul)
VALUES (?, ?, ?)
""", (new_date.strftime("%Y-%m-%d"), new_caucion, new_simul))

conn.commit()
