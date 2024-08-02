import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List, Optional
from gtts import gTTS
import os
from fastapi.responses import FileResponse
import tempfile
from summary import summarize_via_tokenbatches
from pydantic import BaseModel, Field
from Adobesdk.Extract import ExtractTextInfoFromPDF
from dotenv import load_dotenv
load_dotenv()

class Book(BaseModel):
    title: str
    publisher: str

class SummaryRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text to summarize.")
    file: Optional[UploadFile] = Field(None, description="PDF file to extract text from.")

class AudioResponse(BaseModel):
    audio_file: str
    
app = FastAPI()
   
def fetch_books(query):
    api_key = os.getenv(api_key)
    url = f'https://www.googleapis.com/books/v1/volumes?q={query}&key={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  
    else:
        return None


@app.get("/books/search", response_model=List[Book])
async def search_books(query: str):
    books_data = fetch_books(query)
    if books_data:
        books = []
        for item in books_data.get('items', []):
            title = item['volumeInfo'].get('title')
            publisher = item['volumeInfo'].get('publisher', 'Unknown Publisher')
            book = Book(title=title, publisher=publisher)
            books.append(book)
        return books  
    else:
        raise HTTPException(status_code=404, detail="Books not found")
    

@app.post("/generate_audio_summary", response_model=AudioResponse)
async def generate_audio_summary(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    if file and file.content_type == 'application/pdf':
        temp_pdf_path = tempfile.mktemp(suffix=".pdf")
        with open(temp_pdf_path, "wb") as temp_pdf_file:
            temp_pdf_file.write(await file.read())
        extractor = ExtractTextInfoFromPDF(temp_pdf_path)
        pdf_text = extractor.full_text 
        if not pdf_text:
            raise HTTPException(status_code=400, detail="No text found in the PDF.")
        text_to_summarize = pdf_text
    elif text:
        text_to_summarize = text
    else:
        raise HTTPException(status_code=400, detail="Please provide either text or a PDF file.")
    
    summary = summarize_via_tokenbatches(text_to_summarize)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        audio_file_path = temp_audio_file.name
        tts = gTTS(text=summary, lang='en')
        tts.save(audio_file_path)

    return FileResponse(audio_file_path, media_type="audio/mpeg", filename="summary.mp3")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)



