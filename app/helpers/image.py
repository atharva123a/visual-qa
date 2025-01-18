import hashlib
from PIL import Image
import imagehash
from typing import Tuple, List
from fuzzywuzzy import process

from app.db import redis_client

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
    key_prefix = f"hash:{image_hash}" 

    # Retrieve all keys for this hash
    matching_keys = redis_client.keys(f"{key_prefix}:*")
    print(matching_keys, 'matching_keys')
    prompts_in_cache = [key.split(":")[2] for key in matching_keys]

    # Perform fuzzy search to check if prompt exists
    best_match, similarity = process.extractOne(prompt, prompts_in_cache) if prompts_in_cache else (None, 0)
    print(best_match, similarity, 'best_match, similarity')
    if similarity >= 80: 
        print(best_match, 'best_match')
        # Retrieve cached coordinates
        cached_coordinates = redis_client.get(f"{key_prefix}:{best_match}")
        return True, eval(cached_coordinates)

    # Prompt does not exist; generate and store coordinates
    coordinates = detect_coordinates_function(image_path, prompt)

    # Store in Redis as hash:{prompt}
    redis_key = f"{key_prefix}:{prompt}"
    redis_client.set(redis_key, str(coordinates))

    return False, coordinates

