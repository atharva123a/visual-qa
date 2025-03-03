from fastapi import APIRouter, UploadFile, File, Form
import cv2
import pytesseract
from typing import Tuple, List
from app.db import redis_client
from app.utils.cv import detect_coordinates_function
from app.helpers.image import get_or_store_coordinates

router = APIRouter()

@router.post("/process-request/")
async def process_request(file: UploadFile = File(...), instruction: str = Form(...)):
    """Process a request to highlight portion of the image that contains text."""

    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(file.file.read())

    success, coordinates = get_or_store_coordinates(image_path, instruction, detect_coordinates_function)
    
    if(not success):
        return {"message": f"Could not find label for this instruction: {instruction}", "coordinates": []}

    return {"message": "Request processed successfully", "coordinates": coordinates}



