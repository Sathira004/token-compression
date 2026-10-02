from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.v1.api import api_router
from pymongo import AsyncMongoClient

from app.core.config import settings


@asynccontextmanager
async def lifespan(application: FastAPI):
    client = AsyncMongoClient(settings.MONGODB_URI)
    database = client[settings.MONGODB_DATABASE]
    collection = database[settings.MONGODB_RUNS_COLLECTION]
    await client.admin.command("ping")
    await collection.create_index("run_id", unique=True)
    await collection.create_index([("arm", 1), ("created_at", -1)])
    application.state.run_collection = collection
    try:
        yield
    finally:
        await client.close()

app = FastAPI(
    title="Token Compression API",
    description="A modular FastAPI service for compressing text tokens.",
    version="1.0.0",
    lifespan=lifespan,
)

# Root endpoint for health checks
@app.get("/")
def read_root():
    return {"message": "Welcome to the Token Compression API"}

# Include all V1 API routes under the /api/v1 prefix
app.include_router(api_router, prefix="/api/v1")
