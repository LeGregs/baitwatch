"""Web API."""
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from PIL import Image

from baitwatch.settings import FishDetectionEnum
from baitwatch.main import detect_fishes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application's lifespan.

    Put resources that should be initialized once before the app,
    and dropped when closing app.
    """
    # Startup: Initialize resources
    yield
    # Shutdown: Clean up resources

app = FastAPI(lifespan=lifespan)


@app.post("/detect-fishes/")
async def detect(detection_type: FishDetectionEnum, image_file: UploadFile) -> None:
    """Request a fish detection on given image, according to the detection type.

    - detection_type (FishDetectionEnum): Type of detection to use.
    - image_file (UploadFile): image to detect fishes from.
    - Returns: Nothing for now
    """
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))

    return detect_fishes(detection_type, image)


@app.get("/ping/")
async def ping() -> list[str]:
    """PING

    Returns: PONG
    """
    return ["pong"]
