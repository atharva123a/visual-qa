import pytesseract
import easyocr
import os
from PIL import Image

def extract_text_tesseract(image_path: str) -> str:
    """Extracts text from an image using Tesseract."""
    try:
        print(f"Attempting to open image at: {image_path}")
        image = Image.open(image_path)
        # print("Image opened successfully.")
        # image.show()
        text = pytesseract.image_to_string(image)
        print(text, 'text')
        return text
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"Error: {str(e)}"

# print("Current Working Directory:", os.getcwd())
# extract_text_tesseract("../assets/sample_zepto.jpg")


def extract_text_easyocr(image_path: str) -> str:
    """Extracts text from an image using EasyOCR."""
    reader = easyocr.Reader(['en'])
    try:
        # results = reader.readtext(image_path, detail=0)
        # return " ".join(results)
        return "Hello"
    except Exception as e:
        return f"Error: {str(e)}"

extract_text_easyocr("../assets/sample_zepto.jpg")