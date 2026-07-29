from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from .Predict import predict_from_bytes
except ImportError:
    from Predict import predict_from_bytes


app = FastAPI(title="Parastic Skin Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Parastic API running"}


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    image_bytes = await file.read()
    return predict_from_bytes(image_bytes)
