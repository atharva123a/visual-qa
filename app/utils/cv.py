import cv2
from typing import List, Dict, Tuple
import numpy as np
import pytesseract
import os
from sentence_transformers import util
from PIL import Image
from app.db import model


def detect_coordinates_function(image_path: str, instruction: str) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
    """
    Highlights text based on a given instruction and checks if the proper text is highlighted.

    Args:
        image_path (str): Path to the input image.
        instruction (str): Instruction containing the text to highlight.

    Returns:
        Tuple[bool, List[Tuple[int, int, int, int]]]: A boolean indicating if the text was found,
                                                      and a list of bounding boxes for matched text.
    """
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    detected_text = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
    
    target_word = instruction
    print(target_word, 'target_word')
    matches = []
    
    for i, text in enumerate(detected_text['text']):
        text = text.strip()
        if text and len(text) > 1 and target_word.lower() in text.lower():
            x, y, w, h = (detected_text['left'][i], detected_text['top'][i],
                          detected_text['width'][i], detected_text['height'][i])
            matches.append((x, y, w, h))

    for (x, y, w, h) in matches:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        cv2.putText(image, target_word, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Highlighted Text", image)
    cv2.waitKey(6000) 
    cv2.destroyAllWindows()

    # Return success flag and matched bounding boxes
    return len(matches) > 0, matches

def highlight_coordinates(image_path: str, target_word: str, coordinates: List[int]):
    image = cv2.imread(image_path)
    x, y, w, h = coordinates
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    cv2.putText(image, target_word, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Highlighted Text", image)
    cv2.waitKey(6000) 
    cv2.destroyAllWindows()
