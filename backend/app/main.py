from fastapi import FastAPI
from app.routes import mudra

app = FastAPI()

app.include_router(mudra.router)

@app.get("/")
def root():
    return {"message": "NrityaAI backend running"}