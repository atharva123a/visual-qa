from fastapi import FastAPI
import uvicorn
from app.routes import text_extraction
from app.db import return_redis_client

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.
    """
    app = FastAPI()

    # Set up routes
    app.include_router(text_extraction.router)

    return app

# Create the app instance
app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


@app.on_event("shutdown")
async def cleanup_resources():
    print("Shutting down... Cleaning up resources.")
    redis_client = return_redis_client()
    redis_client.close()