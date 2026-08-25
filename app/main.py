from fastapi import FastAPI
from app.api.v1.api import api_router

app = FastAPI(
    title="Token Compression API",
    description="A modular FastAPI service for compressing text tokens.",
    version="1.0.0"
)

# Root endpoint for health checks
@app.get("/")
def read_root():
    return {"message": "Welcome to the Token Compression API"}

# Include all V1 API routes under the /api/v1 prefix
app.include_router(api_router, prefix="/api/v1")
