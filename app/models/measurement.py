from app.database import get_db_connection

class MeasurementModel:
    @staticmethod
    def save(user_id, systolic, diastolic, condition, calibration_id):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO measurements (user_id, systolic, diastolic, condition, calibration_id)
            VALUES (%s, %s, %s, %s, %s) RETURNING id;
            """,
            (user_id, systolic, diastolic, condition, calibration_id)
        )
        m_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return m_id

    @staticmethod
    def get_by_user(user_id):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM measurements WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in rows]