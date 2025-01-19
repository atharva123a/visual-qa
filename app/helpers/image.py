import hashlib
from PIL import Image
import imagehash
from typing import Tuple, List
from app.db import redis_client, model
from app.helpers.helper import get_labels, retreive_best_match, find_label, get_cache, set_cache
from app.utils.cv import highlight_coordinates

# using this to generate unique hash for each image:
def generate_image_hash(image_path):
    # SHA256 for exact matches
    with open(image_path, 'rb') as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    return sha256_hash


# simple function that checks if the hash and prompt exist in Redis. If not, generate hash, detect coordinates, and store it.
def get_or_store_coordinates(
    image_path: str, prompt: str, detect_coordinates_function
) -> Tuple[bool, List[int]]:
    """
    Check if the hash and prompt exist in Redis. If not, generate hash, detect coordinates, and store.
    
    Args:
        image_path (str): Path to the image file.
        prompt (str): The input prompt (e.g., "Tap on search bar").
        detect_coordinates_function: Function to detect coordinates from the image.
    
    Returns:
        Tuple[bool, List[int]]: A success flag and the coordinates.
    """
    # Generate the image hash
    image_hash = generate_image_hash(image_path)

    labels = get_labels(image_hash)
    print(labels, 'labels from redis')
    
    matched_label = retreive_best_match(prompt, labels)

    if(matched_label == 'unknown'):
        print("could not find match inside our redis cache")
        label = find_label(prompt)
        print(label, 'idenitfied this label from instruction')
        success, coordinates = detect_coordinates_function(image_path, label)
        if(not success):
            return False, []
        coordinates = coordinates[0]
        
        set_cache(image_hash, label, coordinates)
        return True, coordinates

    coordinates = get_cache(image_hash, matched_label)
    highlight_coordinates(image_path, matched_label, coordinates)
    return True, coordinates