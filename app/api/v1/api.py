from fastapi import APIRouter
from app.api.v1.routers import naiverag

api_router = APIRouter()

# Include your compression router
api_router.include_router(naiverag.router, prefix="/naive-rag", tags=["naive-rag"])
