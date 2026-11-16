"""Creates a small sample SQLite database for the demo and tests."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

CUSTOMERS = [
    (1, "Asha Rao", "asha.rao@example.com", "Visakhapatnam"),
    (2, "Ravi Kumar", "ravi.k@example.com", "Hyderabad"),
    (3, "Meera Nair", "meera.nair@example.com", "Chennai"),
    (4, "Karthik S", "karthik.s@example.com", "Bengaluru"),
    (5, "Divya Reddy", "divya.r@example.com", "Hyderabad"),
    (6, "Imran Sheikh", "imran.sh@example.com", "Mumbai"),
]

ORDERS = [
    (1, 1, "Laptop Stand", 1499.0, "2026-09-02", "completed"),
    (2, 1, "USB-C Hub", 2199.0, "2026-09-18", "completed"),
    (3, 2, "Mechanical Keyboard", 5499.0, "2026-09-05", "completed"),
    (4, 2, "Webcam", 3299.0, "2026-10-01", "cancelled"),
    (5, 3, "Monitor Arm", 4599.0, "2026-09-12", "completed"),
    (6, 3, "Noise Cancelling Headphones", 12999.0, "2026-10-09", "completed"),
    (7, 4, "Laptop Stand", 1499.0, "2026-10-11", "pending"),
    (8, 4, "USB-C Hub", 2199.0, "2026-10-14", "completed"),
    (9, 5, "Webcam", 3299.0, "2026-10-15", "refunded"),
    (10, 5, "Mechanical Keyboard", 5499.0, "2026-10-20", "completed"),
    (11, 6, "Monitor Arm", 4599.0, "2026-10-22", "cancelled"),
    (12, 6, "Noise Cancelling Headphones", 12999.0, "2026-10-25", "completed"),
    (13, 1, "Webcam", 3299.0, "2026-11-01", "pending"),
    (14, 2, "Laptop Stand", 1499.0, "2026-11-03", "completed"),
    (15, 3, "USB-C Hub", 2199.0, "2026-11-05", "cancelled"),
    (16, 4, "Monitor Arm", 4599.0, "2026-11-07", "completed"),
]


def ensure_db(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT, city TEXT);
            CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, product TEXT,
                                 amount REAL, order_date TEXT, status TEXT);
            """
        )
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?)", CUSTOMERS)
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", ORDERS)
        conn.commit()
    return path
