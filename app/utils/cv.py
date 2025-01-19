import cv2
from typing import List, Dict, Tuple
import numpy as np
from ultralytics import YOLO
import pytesseract
import os
from sentence_transformers import util
from PIL import Image

from app.db import model


def detect_objects(image_path: str, target: str) -> List[Tuple[int, int, int, int]]:

    print('tesserect could not find the target word, so running yolo')
    model = YOLO('yolov8x.pt')


    image = cv2.imread(image_path)

    # Run YOLO on resized dimensions (e.g., 1536x1536)
    results = model(image_path, imgsz=1536, conf=0.7) 

    detections = []  
    detected_results = []
    for result in results:
        for box in result.boxes:
            coordinates = box.xyxy.tolist()[0] 
            confidence = box.conf[0].item()  
            class_id = int(box.cls[0].item())
            label = model.names[class_id] 

            if(label.lower() == target.lower()):
                detected_results.append(result)
                x1, y1, x2, y2 = [int(coordinates[0]), int(coordinates[1]),
                          int(coordinates[2]), int(coordinates[3])]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # returning integer coordinates, because we store them as integers inside redis
                x1, y1, x2, y2 = [int(coordinates[0]), int(coordinates[1]),
                          int(coordinates[2]), int(coordinates[3])]
                detections.append((x1, y1, x2, y2))

    cv2.imshow("Detections", image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    if(len(detections) == 0):
        return False, []

    # returns success, detections, object_detection
    return True, detections, True

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
    if(len(matches) > 0):
        # returns success, matches, object_detection
        return True, matches, False
    
    return detect_objects(image_path, target_word)

def highlight_coordinates(image_path: str, target_word: str, coordinates: List[float], object_detection: bool = False):
    image = cv2.imread(image_path)
    x, y, w, h = coordinates
    if(not object_detection):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        cv2.putText(image, target_word, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Highlighted Text", image)
        cv2.waitKey(6000) 
        cv2.destroyAllWindows()
    else:
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)  # Green box
        cv2.putText(image, f"{target_word}", (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Highlighted Text", image)
        cv2.waitKey(6000) 
        cv2.destroyAllWindows()

