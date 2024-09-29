from fastapi import FastAPI, File, UploadFile
from typing import Union
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import shutil
import os
from random import randbytes

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend
import matplotlib.pyplot as plt

from whisper_gpt import Whisper
from tagging import Tagger
from svaston_detected import ForbiddenDetector
from video_interest import Interester
from object_detection import Detector


VIDEO_DIR = None
FILE_NAME = None
TRANSCRIBITION_TEXT = None

project_path = "/Users/vaneshik/hack/CP_CODE"

process_videos = Whisper(os.path.join(project_path, "models/whisper_small_folder"))

interester = Interester(os.path.join(project_path, "models/VTSum/vtsum_tt.pth"))

forbidden_detector = ForbiddenDetector(
    os.path.join(project_path, "models/YOLOv3_FORBIDDEN/best.pt")
)

tagger = Tagger(
    "fabiochiu/t5-base-tag-generation",
    "/Users/vaneshik/hack/CP_CODE/models/T5_TAGGING_folder",
)

detector = Detector(
    os.path.join(project_path, "models/YOLOv3/yolov3.weights"),
    os.path.join(project_path, "models/YOLOv3/yolov3.cfg"),
    os.path.join(project_path, "models/YOLOv3/coco.names"),
)


app = FastAPI()


@app.get("/")
def root():
    with open("index.html", "r") as f:
        meow = f.read()

    return HTMLResponse(content=meow, status_code=200)


@app.get("/transcribe")
def transcribe():
    global TRANSCRIBITION_TEXT

    if VIDEO_DIR is None:
        return JSONResponse(content={"error": "No video uploaded"}, status_code=400)

    result = process_videos(VIDEO_DIR)
    TRANSCRIBITION_TEXT = next(iter(result.values()))

    return result


@app.get("/tag")
def tag(transcription: Union[str, None] = None):
    if TRANSCRIBITION_TEXT is None and transcription is None:
        return JSONResponse(
            content={
                "error": "transcribe text or provide transcription via get parameter"
            },
            status_code=400,
        )
    tags = tagger(transcription if TRANSCRIBITION_TEXT is None else TRANSCRIBITION_TEXT)
    return tags


@app.get("/detect_object")
def detect_object():
    if FILE_NAME is None:
        return JSONResponse(content={"error": "No video uploaded"}, status_code=400)
    return detector(os.path.join(VIDEO_DIR, FILE_NAME))


@app.get("/detect_forbidden")
def detect_forbidden():
    x = forbidden_detector.predict_bad_symbols(VIDEO_DIR)
    return x


@app.get("/attention_graphic")
def attention_graphic():
    x = interester.get_interests_summary(os.path.join(VIDEO_DIR, FILE_NAME))
    shutil.move("summary.png", os.path.join(VIDEO_DIR, "summary.png"))
    return x[1]


@app.get("/get_all")
def get_all():
    all_data = {
        "transcribe": next(iter(transcribe().values())),
        "tag": ", ".join(sorted(list(set(tag())))),
        "detect_object": ", ".join(
            [x[0].capitalize() for x in detect_object() if x[1] > 10]
        ),
        "detect_forbidden": detect_forbidden()[0][1],
        "attention_graphic": "\n".join(attention_graphic()),
    }

    print(all_data)

    return all_data


# video upload by providing filename
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global VIDEO_DIR
    global FILE_NAME

    try:
        FILE_NAME = file.filename

        upload_dir = "upload-" + randbytes(8).hex()
        VIDEO_DIR = upload_dir

        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return JSONResponse(
            content={"filename": file.filename, "message": "Upload successful"}
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/{filename}.gif")
def get_gif(filename: str):
    file_path = f"{filename}.gif"
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    return FileResponse(file_path)

@app.get("/summary.png")
def get_gif():
    file_path = os.path.join(VIDEO_DIR, "summary.png")
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    return FileResponse(file_path)