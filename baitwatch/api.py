"""Web API."""
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from PIL import Image

from baitwatch.settings import FishDetectionEnum
from baitwatch.main import detect_fishes

from baitwatch.model import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application's lifespan.

    Put resources that should be initialized once before the app,
    and dropped when closing app.
    """
    # Startup: Initialize resources
    app.state.models = {
        FishDetectionEnum.FONF: load_model(),
    }
    yield
    # Shutdown: Clean up resources


app = FastAPI(
    lifespan=lifespan,
    title="Baitwatch API",
    description="Project to detect fishes in photographs.",
    version="1.0.0",
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)


@app.post("/detect-fishes/")
async def detect(detection_type: FishDetectionEnum, image_file: UploadFile):
    """Request a fish detection on given image, according to the detection type.

    - **detection_type** (FishDetectionEnum): Type of detection to use.
    - **image_file** (UploadFile): image to detect fishes from.
    - Returns: Nothing for now
    """
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    model = app.state.models.get(detection_type.value)
    if model is None:
        return {"error": f"No model found for detection type {detection_type.value}"}
    return detect_fishes(model, image)


@app.get("/ping/")
async def ping() -> list[str]:
    """PING

    Returns: PONG
    """
    return ["pong"]
