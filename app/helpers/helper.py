import json
import os
from app.db import redis_client
from typing import List
from autogen import Cache
from openai import OpenAI
from app.config.config import config_load

config = config_load()

openai_client = OpenAI(api_key=config["openai_api_key"])

def set_cache(image_hash: str, label: str, coordinates: List[int], object_detection: bool):
    """
    Set a cache entry in Redis.
    Args:
        image_hash (str): Unique hash of the image.
        label (str): Label extracted from LLM.
        coordinates (List[int]): Coordinates detected from CV function.
        object_detection (bool): True if object detection, False if text detection.
    """
    cache_value = {
        "coordinates": coordinates,
        "object_detection": object_detection
    }
    
    key = f"{image_hash}:{label}"
    value = json.dumps(cache_value)
    redis_client.set(key, value)
    
    print(f"Set cache: {key} -> {value}")

def get_cache(image_hash: str, label: str) -> dict:
    """
    Get a cache entry from Redis.
    """
    key = f"{image_hash}:{label}"
    value = redis_client.get(key) 
    if value:
        print(f"Cache hit: {key} -> {value}")
        return json.loads(value) 
    print(f"Cache miss: {key}")
    return None

def get_labels(image_hash: str) -> List[str]:
    key = f"{image_hash}:*"
    values = redis_client.keys(key)
    labels = [value.split(":")[1] for value in values]
    print(labels, 'labels')
    return labels

def find_label(prompt: str) -> str:
    """
    Find the label for the prompt.
    """
    prompt = prompt.lower()

    system_prompt = f"You are a helpful assistant that extracts the label for a given instruction. A label is a single word that points to a specific part of the image that the instruction is referring to. An example: Tap on the search bar should return search. Toggle the cafe button should return cafe. Return only the label, no other text. If you cannot find the label, return 'unknown'. The prompt is: {prompt}"
    
    response = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user", 
                "content": system_prompt
            }
        ],
        model="gpt-4o",
    )
    return response.choices[0].message.content

def map_instruction_to_label(instruction: str, labels: List[str], cache: Cache):
    """
    Map an instruction to a predefined label using GPT and cache the result.
    
    Args:
        instruction (str): User instruction (e.g., "Tap on search").
        labels (list): List of predefined UI labels (e.g., ["search_bar", "add_button"]).
        cache (Cache): Autogen Redis cache.

    Returns:
        str: Matched label (e.g., "search_bar").
    """
    # Check the cache first
    cached_label = cache.get(f"instruction:{instruction}")
    if cached_label:
        print(f"Cache hit for instruction: {instruction}")
        return cached_label

    # GPT Prompt
    prompt = f"""
    You are an assistant that maps user instructions to predefined UI elements.
    Match the following instruction to one of these labels:
    {', '.join(labels)}
    An example: Tap on the search bar should return search. Toggle the cafe button should return cafe. Return only the label, no other text. If you cannot find the label, return 'unknown'. The prompt is: {instruction}.
    """

    # Make GPT API call
    response = openai_client.chat.completions.create(
         messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ],
        model="gpt-4o",
        max_tokens=10,
        temperature=0,
    )
    label = response.choices[0].message.content
    return label

def retreive_best_match(instruction, labels):
    if(len(labels) == 0):
        print("could not find label")
        return "unknown"

    with Cache.redis(redis_url="redis://localhost:6379/0") as cache:
        matched_label = map_instruction_to_label(instruction, labels, cache)
        print(f"Matched Label: {matched_label}")

    return matched_label