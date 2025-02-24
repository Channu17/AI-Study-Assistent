from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from src.utils.bot import resume, get_model, initialize_retriver, initialize_rag_chain
from dotenv import load_dotenv
import os
from src.utils.database import insert_application_logs, get_chat_history
import uuid
from typing import Optional
from pdf2image import convert_from_bytes
from io import BytesIO
from PyPDF2 import PdfMerger
import pytesseract
import traceback

load_dotenv()

app = FastAPI()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

model,embeddings = get_model()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # for windows

@app.get("/")
def helloworld():
    return {"message": "Hello World"}


@app.post("/resumeAnalyser")
async def resumeAnalyser(file: UploadFile = File(...), question: str = ""):
    return {"answer": resume(file, question)}


@app.post("/chat")
def chat(session_id: Optional[str] = None, user_query: str = "", subject: str = "Data Communication", sem: int = 4):
    _, retriever = initialize_retriver(model, embeddings, subject, sem)
    rag_chain = initialize_rag_chain(model, retriever)
    if not session_id:
        session_id = str(uuid.uuid4())
    try:
        chat_history = get_chat_history(session_id)
        response = rag_chain.invoke({
        "input": user_query, "chat_history": chat_history, "subject":subject})['answer']

        insert_application_logs(session_id, user_query, response)
        return {"session_id": session_id, "response": response}
    except KeyError as e:
        raise HTTPException(status_code=500, detail =f"Error Processing th eresponse :{e}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occured: {e}")


@app.post("/OCR")
async def convert_to_searchable_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
        pdf_bytes = await file.read()
        
        images = convert_from_bytes(pdf_bytes, dpi=500)
        if not images:
            raise HTTPException(status_code=400, detail="Failed to extract images from the PDF")
        
        ocr_pdfs = []
        for img in images:
            try:
                pdf_data = pytesseract.image_to_pdf_or_hocr(img, extension='pdf')
                ocr_pdfs.append(BytesIO(pdf_data))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
            
        if not ocr_pdfs:
            raise HTTPException(status_code=500, detail="OCR conversion resulted in an empty file")
        
        merger = PdfMerger()
        for ocr_pdf in ocr_pdfs:
            merger.append(ocr_pdf)
        
        output_pdf = BytesIO()
        merger.write(output_pdf)
        merger.close()
        output_pdf.seek(0)
        
        original_filename = file.filename.rsplit(".", 1)[0] if file.filename else "output"
        searchable_filename = f"{original_filename}_searchable.pdf"
        
        return StreamingResponse(output_pdf, media_type="application/pdf", headers={
            "Content-Disposition": f"attachment; filename={searchable_filename}"
        })
    except HTTPException as http_err:
        raise http_err  
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
