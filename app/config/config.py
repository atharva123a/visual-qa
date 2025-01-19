import os
from dotenv import load_dotenv

load_dotenv()

def config_load():
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    return {
        "openai_api_key": openai_api_key
    }