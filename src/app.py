import streamlit as st
import requests
import uuid

# FastAPI backend URL
API_BASE_URL = "https://ai-study-assistent.onrender.com"

st.set_page_config(page_title="AI Assistant", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Study Chatbot"])

if page == "Study Chatbot":
    st.title("Study Chatbot")
    subjects = ["Data Communication", "Design and Analysis of Algorithms", "Linear Algebra", "Operating Systems", "Software Engineering", "Theory of Computation"]
    subject = st.selectbox("Select Subject", subjects, index=0)
    sem = st.number_input("Enter Semester", min_value=1, max_value=10, value=4, step=1)
    user_query = st.text_area("Enter your question:")
    
    if st.button("Get Answer"):
        session_id = str(uuid.uuid4())
        payload = {"session_id": session_id, "user_query": user_query, "subject": subject, "sem": sem}
        response = requests.post(f"{API_BASE_URL}/chat", json=payload)
        if response.status_code == 200:
            st.write("### Chatbot Response:", response.json()["response"])
        else:
            st.error("Failed to fetch response from chatbot")