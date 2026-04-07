import duckdb

# Connect to a file-based database (or ':memory:' for RAM-only)
con = duckdb.connect('my_local_db.duckdb')

with open('data.csv', 'w') as f:
    f.write("id,item,price\n1,keyboard,50\n2,mouse,25\n3,monitor,200")

# Query the CSV file directly like it's a table
query = """
SELECT item, price * 1.1 AS price_with_tax 
FROM 'data.csv' 
WHERE price > 30
"""

print(duckdb.query(query).to_df())