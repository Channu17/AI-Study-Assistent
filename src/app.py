import streamlit as st
from utils.bot import resume, get_model, initialize_retriver, initialize_rag_chain
from dotenv import load_dotenv
import os
from utils.database import insert_application_logs, get_chat_history
import uuid

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Load model and embeddings
model, embeddings = get_model()

# Streamlit UI
st.set_page_config(page_title="AI Assistant", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Resume Analyzer", "Study Chatbot"])

if page == "Resume Analyzer":
    st.title("Resume Analyzer")
    uploaded_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])
    
    if uploaded_file:
        if st.button("ATS Score"):
            ats_score = resume(uploaded_file, "ats_score")
            st.write("### ATS Score:", ats_score)
        
        if st.button("Grammatical Mistakes"):
            grammar_mistakes = resume(uploaded_file, "grammar_mistakes")
            st.write("### Grammatical Mistakes:", grammar_mistakes)
        
        if st.button("Improvements"):
            improvements = resume(uploaded_file, "improvements")
            st.write("### Suggested Improvements:", improvements)
        
        if st.button("Suggestions"):
            suggestions = resume(uploaded_file, "suggestions")
            st.write("### Suggestions:", suggestions)
    else:
        st.warning("Please upload a resume file.")

elif page == "Study Chatbot":
    st.title("Study Chatbot")
    subjects = ["Data Communication", "Design and Analysis of Algorithms", "Linear Algebra", "Operating Systems", "Software Engineering", "Theory of Computation"]
    subject = st.selectbox("Select Subject", subjects, index=0)
    sem = st.number_input("Enter Semester", min_value=1, max_value=10, value=4, step=1)
    user_query = st.text_area("Enter your question:")
    if st.button("Get Answer"):
        session_id = str(uuid.uuid4())
        _, retriever = initialize_retriver(model, embeddings, subject, sem)
        rag_chain = initialize_rag_chain(model, retriever)
        chat_history = get_chat_history(session_id)
        response = rag_chain.invoke({
            "input": user_query, "chat_history": chat_history, "subject": subject
        })['answer']
        insert_application_logs(session_id, user_query, response)
        st.write("### Chatbot Response:", response)
