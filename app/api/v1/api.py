from fastapi import APIRouter
from app.api.v1.routers import agent

api_router = APIRouter()

# Register agent routing triggers under an identifiable /agent workspace scope
api_router.include_router(agent.router, prefix="/agent", tags=["Agent Workflows"])
