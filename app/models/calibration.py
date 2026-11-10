from app.database import get_db_connection
import psycopg2.extras

class CalibrationModel:
    @staticmethod
    def save(user_id, systolic_ref, diastolic_ref, signal_features):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO calibrations (user_id, systolic_ref, diastolic_ref, signal_features)
            VALUES (%s, %s, %s, %s) RETURNING id;
            """,
            (user_id, systolic_ref, diastolic_ref, psycopg2.extras.Json(signal_features))
        )
        cal_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return cal_id

    @staticmethod
    def get_by_user(user_id):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM calibrations WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in rows]

    @staticmethod
    def get_latest(user_id):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM calibrations WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else None