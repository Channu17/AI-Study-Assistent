# AI Study Assistant 🎓

An intelligent study companion powered by RAG (Retrieval-Augmented Generation) technology that helps engineering students with their coursework. The assistant provides accurate, context-aware answers based on textbook content and includes additional features like resume analysis and OCR capabilities.

## 🌐 Live Deployment

- **API**: [https://ai-study-assistent.onrender.com](https://ai-study-assistent.onrender.com)
- **Web App**: [https://ai-study-assistent.streamlit.app/](https://ai-study-assistent.streamlit.app/)

## 🚀 Features

### 📚 Study Chatbot
- **Subject-specific Q&A**: Get answers from textbook content for various engineering subjects
- **Context-aware responses**: Maintains chat history for coherent conversations
- **Multi-semester support**: Currently supports semester 4 with 6 subjects
- **Academic accuracy**: Answers are strictly based on provided textbook content

### 📄 Resume Analyzer
- **ATS Score Analysis**: Get your resume's ATS compatibility score
- **Document Processing**: Supports PDF and DOCX formats
- **Intelligent Feedback**: Provides suggestions when requested

### 🔍 OCR Service
- **Searchable PDF Creation**: Convert scanned PDFs to searchable format
- **High-quality processing**: Uses Tesseract OCR with 500 DPI for optimal results
- **Batch processing**: Handles multi-page documents efficiently

## 📋 Supported Subjects (Semester 4)

- Data Communication
- Design and Analysis of Algorithms
- Linear Algebra
- Operating Systems
- Software Engineering
- Theory of Computation

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance web framework
- **LangChain**: LLM application framework
- **FAISS**: Vector database for similarity search
- **Groq**: LLM provider (Llama-3.3-70b-versatile)
- **HuggingFace**: Embeddings (all-MiniLM-L6-v2)

### Frontend
- **Streamlit**: Interactive web application framework

### Database
- **SQLite**: Session management and chat history storage

### OCR & Document Processing
- **Tesseract**: Optical Character Recognition
- **PDF2Image**: PDF to image conversion
- **PyPDF2**: PDF manipulation
- **python-docx**: Word document processing

## 📁 Project Structure

```
Study-Sync-Assistent/
├── src/
│   ├── api.py                 # FastAPI backend
│   ├── app.py                 # Streamlit frontend
│   ├── RAG/                   # Vector databases by semester/subject
│   │   └── sem4/
│   │       ├── Data Communication/
│   │       ├── Design and Analysis of Algorithms/
│   │       └── ...
│   ├── TextBooks/             # Source textbooks
│   │   └── sem4/
│   ├── utils/
│   │   ├── bot.py            # RAG chain and model initialization
│   │   ├── database.py       # SQLite operations
│   │   └── extractor.py      # Document text extraction
│   └── uploads/              # Temporary file storage
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Tesseract OCR (for Windows: install from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki))

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Study-Sync-Assistent
```

### 2. Create Virtual Environment
```bash
python -m venv studysync-env
source studysync-env/bin/activate  # On Windows: studysync-env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### 5. Configure Tesseract (Windows)
Update the tesseract path in `src/api.py` if different:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 🚀 Running the Application

### Backend API
```bash
cd src
python api.py
# Or using uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Frontend Application
```bash
streamlit run src/app.py
```

## 📖 API Documentation

### Endpoints

#### `GET /`
Health check endpoint
- **Response**: `{"message": "Hello World"}`

#### `POST /resumeAnalyser`
Analyze resume and answer questions
- **Parameters**: 
  - `file`: Resume file (PDF/DOCX)
  - `question`: Question about the resume
- **Response**: `{"answer": "analysis result"}`

#### `POST /chat`
Study chatbot endpoint
- **Parameters**:
  - `session_id`: Optional session identifier
  - `user_query`: Student's question
  - `subject`: Subject name (default: "Data Communication")
  - `sem`: Semester number (default: 4)
- **Response**: `{"session_id": "uuid", "response": "answer"}`

#### `POST /OCR`
Convert scanned PDF to searchable PDF
- **Parameters**: 
  - `file`: Scanned PDF file
- **Response**: Searchable PDF file download

## 🔒 Security Features

- Input validation for all file uploads
- Error handling with appropriate HTTP status codes
- Session-based chat history management
- Temporary file cleanup after processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Known Issues

- OCR processing may take time for large documents
- Currently supports only semester 4 subjects
- Tesseract path needs manual configuration on different systems

## 🔮 Future Enhancements

- [ ] Support for additional semesters
- [ ] Multi-language support
- [ ] Real-time streaming responses
- [ ] Enhanced UI/UX
- [ ] Mobile application
- [ ] Advanced analytics dashboard

## 📧 Contact

For questions, suggestions, or support, please open an issue in the repository or contact the development team.

---

**Made with ❤️ for engineering students**
