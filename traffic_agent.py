import sqlite3
import os
import json
from datetime import datetime

# You would normally import your LLM library here, e.g.
# import google.generativeai as genai

class TrafficAgent:
    def __init__(self, db_path):
        self.db_path = db_path
        # self.api_key = os.getenv("GEMINI_API_KEY") 
        # if self.api_key:
        #     genai.configure(api_key=self.api_key)
        #     self.model = genai.GenerativeModel('gemini-pro')
        
    def query_db(self, query, params=()):
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

    def analyze(self, user_query):
        """
        Simulates an AI Agent analyzing traffic data.
        In a real scenario, this would use an LLM to convert user_query -> SQL.
        For now, we use keyword matching to simulate 'intelligence'.
        """
        user_query = user_query.lower()
        
        response = "I'm not sure how to answer that yet."
        data = None

        if "how many" in user_query or "count" in user_query:
            if "vehicle" in user_query:
                data = self.query_db("SELECT COUNT(*) as count FROM vehicle_logs")
                response = f"I found {data[0]['count']} vehicles in the logs."
            
            if "speeding" in user_query or "violation" in user_query:
                data = self.query_db("SELECT COUNT(*) as count FROM vehicle_logs WHERE is_speeding=1")
                response = f"There have been {data[0]['count']} speeding violations recorded."

        elif "fastest" in user_query or "max speed" in user_query:
            data = self.query_db("SELECT MAX(speed) as max_speed, type, plate_text FROM vehicle_logs")
            if data and data[0]['max_speed']:
                response = f"The fastest vehicle was a {data[0]['type']} (Plate: {data[0]['plate_text']}) going {round(data[0]['max_speed'], 1)} km/h."
            else:
                response = "No speed data available yet."

        elif "average" in user_query and "speed" in user_query:
            data = self.query_db("SELECT AVG(speed) as avg_speed FROM vehicle_logs")
            val = round(data[0]['avg_speed'], 1) if data[0]['avg_speed'] else 0
            response = f"The average traffic speed is {val} km/h."

        elif "helmet" in user_query:
             # Placeholder for future helmet stats
             response = "I am tracking helmet usage, but no violations have been flagged in the database yet (Feature In Progress)."

        else:
            # Fallback for "General" chat
            response = f"I can help you analyze traffic data. Try asking: 'How many speeding vehicles?', 'What is the average speed?', or 'Show me the fastest car'."

        return {
            "response": response,
            "data_context": data # optionally return raw data for charts
        }
