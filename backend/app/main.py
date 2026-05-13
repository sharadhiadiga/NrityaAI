from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chatbot

#for community
from app.database import engine
from app.models import community as community_models
from app.routes import community as community_routes

app = FastAPI(
    title="Nritya AI",
    description="AI-powered dance analysis and recognition API",
    version="0.1.0",
)

community_models.Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(community_routes.router)
app.include_router(chatbot.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Nritya AI API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
