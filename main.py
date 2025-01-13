import streamlit as st
import uuid
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')


model = ChatGroq(groq_api_key=groq_api_key, model='llama-3.3-70b-versatile')
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')


vector_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever()

contextualize_q_system_prompt = '''
You are a AI study assistant for the Enginnering students
    - Dont ask back any questions.
    - directly answer the questions without greeting qestions
    - Answer the question with respect to text book only dont answer if not there
    - dont answer irrevelent questions
    - Answer every qestions for about 200-250 words if possible'''

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ('human', '{input}')
    ]
)

history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ('system', '''
You are a AI study assistant for the Enginnering students
    - Dont ask back any questions.
    - directly answer the questions without greeting qestions
    - Answer the question with respect to text book only dont answer if not there
    - dont answer irrevelent questions
    - Answer every qestions for about 200-250 words if possible
    - context: {context}'''),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_chat_history(session_id):
    """Retrieve chat history for a specific session."""
    if session_id not in st.session_state:
        st.session_state[session_id] = []
    return st.session_state[session_id]

def insert_application_logs(session_id, user_query, response):
    """Log user query and AI response to session history."""
    if session_id not in st.session_state:
        st.session_state[session_id] = []
    st.session_state[session_id].append({"query": user_query, "response": response})

st.title("AI Study Assistant for Engineering Students")
st.write("Ask any question from your textbooks, and get accurate and contextual answers.")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

session_id = st.session_state["session_id"]

user_query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("Fetching answer..."):
            try:
               
                chat_history = get_chat_history(session_id)
                
        
                response = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": chat_history
                })['answer']
      
                insert_application_logs(session_id, user_query, response)

                st.write("### Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid question.")
        
if st.checkbox("Show Chat History"):
    st.write("### Chat History")
    history = get_chat_history(session_id)
    for idx, entry in enumerate(history):
        st.write(f"**Q{idx+1}:** {entry['query']}")
        st.write(f"**A{idx+1}:** {entry['response']}")
