"""Duetify API. Run: uvicorn app:app --reload  (from the backend folder)"""
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

import pipeline

app = FastAPI(title="Duetify")
ALLOWED = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


async def save(upload: UploadFile):
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use wav, mp3, m4a, ogg or flac.")
    fd, path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(await upload.read())
    return path


@app.post("/api/transcribe")
async def transcribe(
    part1: UploadFile = File(...),
    inst1: str = Form("violin"),
    part2: UploadFile | None = File(None),
    inst2: str = Form("piano"),
    level: str = Form("normal"),
    bpm: int | None = Form(None),
):
    if level not in pipeline.LEVELS:
        raise HTTPException(400, "level must be 'normal' or 'simple'")
    for i in (inst1, inst2):
        if i not in pipeline.RANGES:
            raise HTTPException(400, f"Unknown instrument '{i}'")
    paths = []
    try:
        paths.append((await save(part1), inst1))
        if part2 is not None and part2.filename:
            paths.append((await save(part2), inst2))
        return pipeline.run(paths, bpm=bpm, level=level)
    except ValueError as e:  # our own friendly errors from the pipeline
        raise HTTPException(422, str(e))
    finally:
        for p, _ in paths:
            os.remove(p)


# serve the web page from ../frontend
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "frontend", html=True), name="web")
