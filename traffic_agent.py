import sqlite3
import os

# Gemini API integration (requires GEMINI_API_KEY env variable)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class TrafficAgent:
    def __init__(self, db_path):
        self.db_path = db_path
        self.gemini_model = None

        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-pro")
                print("✅ Gemini AI Agent connected successfully.")
            except Exception as e:
                print(f"⚠️  Gemini setup failed: {e}")
        else:
            print("ℹ️  Gemini API key not found. Using keyword-based agent fallback.")

    def query_db(self, query, params=()):
        """Execute a SQL query and return results as list of dicts."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute(query, params)
            rows = c.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            return f"Database Error: {e}"

    def get_db_summary(self):
        """Fetch a brief summary of the DB for LLM context."""
        try:
            total = self.query_db("SELECT COUNT(*) as c FROM vehicle_logs")[0]['c']
            speeding = self.query_db("SELECT COUNT(*) as c FROM vehicle_logs WHERE is_speeding=1")[0]['c']
            avg_speed = self.query_db("SELECT ROUND(AVG(speed),1) as a FROM vehicle_logs")[0]['a'] or 0
            max_speed = self.query_db("SELECT ROUND(MAX(speed),1) as m, type, plate_text FROM vehicle_logs")[0]
            types_raw = self.query_db("SELECT type, COUNT(*) as cnt FROM vehicle_logs GROUP BY type ORDER BY cnt DESC")
            types_str = ", ".join([f"{r['type']}={r['cnt']}" for r in types_raw]) or "none"
            summary = (
                f"Traffic DB Summary: total={total} vehicles, speeding_violations={speeding}, "
                f"avg_speed={avg_speed}km/h, max_speed={max_speed['m']}km/h "
                f"(a {max_speed['type']}, plate:{max_speed['plate_text']}), vehicle_types=[{types_str}]."
            )
            return summary
        except Exception as e:
            return f"Could not summarise DB: {e}"

    def analyze_with_gemini(self, user_query):
        """Use Gemini to answer the user query using DB context."""
        db_summary = self.get_db_summary()
        prompt = (
            "You are an AI Traffic Analyst assistant. You have access to the following real-time traffic database summary:\n\n"
            f"{db_summary}\n\n"
            "Answer the following user question concisely and helpfully. "
            "If numbers come from the summary, use them. "
            "If the query is outside traffic scope, politely redirect.\n\n"
            f"User: {user_query}"
        )
        try:
            response = self.gemini_model.generate_content(prompt)
            return {"response": response.text, "data_context": None}
        except Exception as e:
            return {"response": f"Gemini error: {e}. Falling back to keyword analysis.", "data_context": None}

    def analyze(self, user_query):
        """
        Main analysis method. Uses Gemini if available, else falls back to keyword matching.
        """
        # --- Gemini Path ---
        if self.gemini_model:
            return self.analyze_with_gemini(user_query)

        # --- Keyword Fallback Path ---
        q = user_query.lower()
        response = "I can help you analyze traffic data. Try: 'How many speeding vehicles?', 'What is the average speed?', 'Show fastest car', 'Count vehicles by type'."
        data = None

        if "how many" in q or "count" in q:
            if "speeding" in q or "violation" in q:
                data = self.query_db("SELECT COUNT(*) as count FROM vehicle_logs WHERE is_speeding=1")
                response = f"There have been {data[0]['count']} speeding violations recorded."
            elif "vehicle" in q:
                data = self.query_db("SELECT COUNT(*) as count FROM vehicle_logs")
                response = f"I found {data[0]['count']} total vehicles in the logs."
            elif "car" in q:
                data = self.query_db("SELECT COUNT(*) as count FROM vehicle_logs WHERE type='car'")
                response = f"There are {data[0]['count']} cars in the logs."
            elif "truck" in q or "bus" in q:
                vtype = "truck" if "truck" in q else "bus"
                data = self.query_db("SELECT COUNT(*) as count FROM vehicle_logs WHERE type=?", (vtype,))
                response = f"There are {data[0]['count']} {vtype}s in the logs."

        elif "fastest" in q or "max speed" in q:
            data = self.query_db("SELECT MAX(speed) as max_speed, type, plate_text FROM vehicle_logs")
            if data and data[0]['max_speed']:
                response = f"The fastest vehicle was a {data[0]['type']} (Plate: {data[0]['plate_text']}) going {round(data[0]['max_speed'], 1)} km/h."
            else:
                response = "No speed data available yet."

        elif "average" in q and "speed" in q:
            data = self.query_db("SELECT AVG(speed) as avg_speed FROM vehicle_logs")
            val = round(data[0]['avg_speed'], 1) if data and data[0]['avg_speed'] else 0
            response = f"The average traffic speed is {val} km/h."

        elif "slowest" in q or "min speed" in q:
            data = self.query_db("SELECT MIN(speed) as min_speed, type, plate_text FROM vehicle_logs WHERE speed > 0")
            if data and data[0]['min_speed']:
                response = f"The slowest vehicle was a {data[0]['type']} (Plate: {data[0]['plate_text']}) going {round(data[0]['min_speed'], 1)} km/h."
            else:
                response = "No data available yet."

        elif "helmet" in q:
            data = self.query_db("SELECT COUNT(*) as count FROM vehicle_logs WHERE is_helmet_missing=1")
            response = f"There are {data[0]['count']} helmet violation(s) recorded in the database."

        elif "lane" in q:
            data = self.query_db("SELECT lane, COUNT(*) as count FROM vehicle_logs GROUP BY lane")
            if data:
                lanes = ", ".join([f"Lane {r['lane']}: {r['count']} vehicles" for r in data])
                response = f"Lane distribution — {lanes}."
            else:
                response = "No lane data yet."

        elif "type" in q or "breakdown" in q or "summary" in q:
            data = self.query_db("SELECT type, COUNT(*) as count FROM vehicle_logs GROUP BY type ORDER BY count DESC")
            if data:
                breakdown = ", ".join([f"{r['type']}s: {r['count']}" for r in data])
                response = f"Vehicle type breakdown — {breakdown}."
            else:
                response = "No vehicle type data available yet."

        return {"response": response, "data_context": data}
