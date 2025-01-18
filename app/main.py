from fastapi import FastAPI
from app.routers import text_extraction
import uvicorn
app = FastAPI()

app.include_router(text_extraction.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)