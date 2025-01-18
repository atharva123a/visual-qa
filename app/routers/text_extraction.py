from fastapi import APIRouter, UploadFile, File
from app.utils.cv import extract_text_tesseract, extract_text_easyocr

router = APIRouter()

@router.post("/extract-text/")
async def extract_text(file: UploadFile = File(...), method: str = "tesseract"):
    """Extract text from an uploaded image."""
    try:
        image_path = f"/tmp/{file.filename}"
        with open(image_path, "wb") as f:
            f.write(file.file.read())
        
        if method == "tesseract":
            text = extract_text_tesseract(image_path)
        elif method == "easyocr":
            text = extract_text_easyocr(image_path)
        else:
            return {"error": "Unsupported method"}

        return {"text": text}
    except Exception as e:
        return {"error": str(e)}
