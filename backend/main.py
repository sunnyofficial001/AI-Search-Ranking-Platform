import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as api_router

app = FastAPI(
    title="Amazon-grade Learning-to-Rank (LTR) Retrieval & Recs API",
    description="Production-grade, fully runnable FastAPI search indexing & hybrid recommendation engine.",
    version="1.0.0",
)

# Configure CORS so that our React web application can call it seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root_endpoint():
    return {
        "message": "Welcome to the Amazon & Netflix-grade Learning-to-Rank Recommendation System API Console.",
        "documentation": "/docs",
        "health": "/api/v1/health",
    }


# Register all API endpoints
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
