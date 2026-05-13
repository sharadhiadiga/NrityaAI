import asyncio
import logging
import os
from contextlib import asynccontextmanager

# Quiet TensorFlow / absl before any app import pulls mediapipe.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import mudra


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        mudra.load_model()
    except FileNotFoundError:
        # Allow server to start; predict will try again or fail clearly.
        pass
    yield
    # Run teardown in a thread so a slow MediaPipe close does not block shutdown.
    try:
        await asyncio.to_thread(mudra.unload_artifacts)
    except asyncio.CancelledError:
        mudra.unload_artifacts()
        raise


app = FastAPI(title="NrityaAI Mudra", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(mudra.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
