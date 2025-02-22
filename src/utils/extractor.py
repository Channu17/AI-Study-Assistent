import pypdf
import docx2txt

def pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def docx_to_text(docx_path):
    return docx2txt.process(docx_path)

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return pdf_to_text(file_path)
    elif file_path.endswith(".docx"):
        return docx_to_text(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    