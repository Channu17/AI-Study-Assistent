from pymongo import MongoClient
from datetime import datetime

DB_NAME = "rag_app"
COLLECTION_NAME = "application_logs"


def get_db_connection():
    client = MongoClient("mongodb://localhost:27017/")  
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


def insert_application_logs(session_id, user_query, model_response):
    collection = get_db_connection()
    log_entry = {
        "session_id": session_id,
        "user_query": user_query,
        "model_response": model_response,
        "created_at": datetime.utcnow()
    }
    collection.insert_one(log_entry)
    
  
    logs = list(collection.find({"session_id": session_id}).sort("created_at", -1))
    if len(logs) > 3:
        for log in logs[3:]:
            collection.delete_one({"_id": log["_id"]})

def get_chat_history(session_id):
    collection = get_db_connection()
    logs = collection.find({"session_id": session_id}, {"_id": 0, "user_query": 1, "model_response": 1}).sort("created_at", -1).limit(3)
    
    messages = []
    for log in logs:
        messages.extend([
            {"role": "human", "content": log["user_query"]},
            {"role": "ai", "content": log["model_response"]}
        ])
    
    return messages
