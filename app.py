from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Multimodal Video Dubbing - API running"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # Save upload to a temporary folder and print filename to console
    tmp_dir = Path("tmp_uploads")
    tmp_dir.mkdir(exist_ok=True)
    dest = tmp_dir / file.filename

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"Uploaded file: {file.filename}")
    return JSONResponse({"filename": file.filename, "path": str(dest)})