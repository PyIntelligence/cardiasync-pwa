import psycopg2
import psycopg2.extras
from flask import current_app

def get_db_connection():
    """Retorna uma conexão com o PostgreSQL (Neon)"""
    conn = psycopg2.connect(
        current_app.config['DATABASE_URL'],
        cursor_factory=psycopg2.extras.DictCursor
    )
    return conn