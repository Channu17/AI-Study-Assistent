from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.utils.extractor import extract_text
import os
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
load_dotenv()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

def get_model():
    groq_api_key = os.getenv("GROQ_API_KEY")
    model = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return model, embeddings

def resume(file: UploadFile, question: str):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    model,_ = get_model()
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
            
        resume_text = extract_text(file_path)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Extracted resume text is empty or invalid.")
        
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
        question_prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant analyzing a resume. Answer concisely based on the resume content.
            - If asked for the ATS score, provide only the score in percentage along with a brief assessment.
            - Provide suggestions only if explicitly requested.

            Resume: {resume_text}
            
            Question: {question}"""
        )
        
        formatted_prompt = question_prompt.format(resume_text=resume_text, question=question)
        response = model.invoke(formatted_prompt)        
        return response.content.strip()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def initialize_retriver(model, embeddings, subject, sem):
    vector_db = FAISS.load_local(f'src/RAG/sem{sem}/{subject}', embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()

    contextualize_q_system_prompt = """
    You are an AI study assistant for engineering students, answering only from the provided textbook.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )
    
    return history_aware_retriever, retriever

def initialize_rag_chain(model, retriever):
    qa_prompt = ChatPromptTemplate.from_messages([
    ('system', '''You are an AI study assistant for{subject}, answering only from the provided textbook. 
    Give clear, concise, academic responses without speculation or AI-like phrasing. Structure answers well.
    Dont answer the same question twice, dont answer questions that are not in the textbook, and dont provide answer that are from other subjects.
    -context: {context}'''),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

